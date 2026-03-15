#!/usr/bin/env python3
"""
DANP (Decorrelation-adapted Node Perturbation) for finetuning Qwen 0.5B on conciseness task.

Uses activation noise + loss difference (L_noisy - L_clean) for gradient estimation.
Hyperparameters (from toy experiments): eta=1e-3, sigma=1e-3, alpha=1e-4, no scaling.

Usage:
  python danp_llm_conciseness.py --n_train 32 --n_eval 50 --epochs 5 --batch_size 2
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import os
import argparse
import re
import hashlib
from datasets import load_dataset
from tqdm import tqdm

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

ETA, SIGMA, ALPHA = 1e-3, 1e-3, 1e-4

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='./out_danp_conciseness')
parser.add_argument('--n_train', type=int, default=64)
parser.add_argument('--n_eval', type=int, default=50)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--eta', type=float, default=ETA)
parser.add_argument('--sigma', type=float, default=SIGMA)
parser.add_argument('--alpha', type=float, default=ALPHA)
parser.add_argument('--np_include', type=str, default='mlp', choices=['head','attn','mlp','all'])
parser.add_argument('--last_k', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--eval_interval', type=int, default=1)
args = parser.parse_args()


def _stable_uid(s: str) -> int:
    h = hashlib.blake2s(s.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(h[:4], 'little') & 0x7FFFFFFF


def _infer_total_blocks(model) -> int:
    names = [n for n, _ in model.named_modules() if re.search(r'\.layers\.(\d+)\.', n)]
    if not names:
        return 0
    return max(int(re.search(r'\.layers\.(\d+)\.', n).group(1)) for n in names) + 1


def _is_target_linear(include: str, name: str, mod, last_k: int, total_blocks: int) -> bool:
    if not isinstance(mod, torch.nn.Linear):
        return False
    lname = name.lower()
    if last_k > 0 and total_blocks > 0:
        m = re.search(r'\.layers\.(\d+)\.', name)
        if m and int(m.group(1)) < total_blocks - last_k:
            return False
    if include == 'head':
        return lname.endswith('lm_head')
    if include == 'attn':
        return any(k in lname for k in ('attn','attention','q_proj','k_proj','v_proj','o_proj','in_proj'))
    if include == 'mlp':
        return any(k in lname for k in ('mlp','ffn','gate_proj','up_proj','down_proj','fc1','fc2'))
    return True


def _named_linear_params(model, include, last_k):
    total_blocks = _infer_total_blocks(model)
    for name, mod in model.named_modules():
        if _is_target_linear(include, name, mod, last_k, total_blocks):
            if getattr(mod, "weight", None) is not None:
                yield name + ".weight", mod.weight
            if getattr(mod, "bias", None) is not None:
                yield name + ".bias", mod.bias


class DANPActivationHook:
    def __init__(self, model, include, last_k, sigma, base_seed):
        self.model, self.include, self.last_k = model, include, last_k
        self.sigma, self.base_seed = float(sigma), int(base_seed)
        self._orig, self._targets = {}, []
        self._row_counters, self._layer_uids = {}, {}
        self._track, self._grad_scale, self._add_noise = False, 1.0, True
        self._total_blocks = _infer_total_blocks(model)
        self.norm_sq, self.total_N = 0.0, 0
        for name, mod in model.named_modules():
            if _is_target_linear(include, name, mod, last_k, self._total_blocks):
                self._targets.append((name, mod))
        for name, _ in self._targets:
            self._row_counters[name] = 0
            self._layer_uids[name] = _stable_uid(name)

    def _seed_for_call(self, layer_uid: int, row_start: int) -> int:
        x = (self.base_seed ^ layer_uid) + 0x9e3779b9 + (row_start * 2654435761 & 0x7FFFFFFF)
        return x & 0x7FFFFFFF

    def attach(self, add_noise: bool = True, track: bool = False, grad_scale: float = 1.0):
        self._add_noise, self._track, self._grad_scale = add_noise, track, float(grad_scale)
        self.norm_sq, self.total_N = 0.0, 0
        for name, mod in self._targets:
            self._row_counters[name] = 0
        for name, mod in self._targets:
            self._wrap(name, mod)

    def detach(self):
        for name, mod in self._targets:
            if name in self._orig:
                mod.forward = self._orig[name]
        self._orig.clear()

    def _wrap(self, name, mod: torch.nn.Linear):
        orig, layer_uid = mod.forward, self._layer_uids[name]
        def fwd(x):
            x_in = x
            if x_in.dim() > 2:
                N, C = int(np.prod(x_in.shape[:-1])), x_in.shape[-1]
                x2, reshape_back = x_in.reshape(N, C), True
            else:
                x2, N, C, reshape_back = x_in, x_in.shape[0], x_in.shape[1], False
            with torch.no_grad():
                y = orig(x2)
                y_dtype, y32 = y.dtype, y.float()
                out_dim, dev = y32.shape[-1], y32.device
                g = torch.Generator(device=dev)
                g.manual_seed(self._seed_for_call(layer_uid, self._row_counters[name]))
                xi = torch.randn((N, out_dim), generator=g, device=dev, dtype=torch.float32) * self.sigma
                if self._add_noise:
                    self.norm_sq += (xi ** 2).sum().item()
                    self.total_N += N * out_dim
                if self._track:
                    if mod.weight.grad is None:
                        mod.weight.grad = torch.zeros_like(mod.weight)
                    mod.weight.grad.add_((self._grad_scale * xi.t().mm(x2.float())).to(mod.weight.dtype))
                    if mod.bias is not None:
                        if mod.bias.grad is None:
                            mod.bias.grad = torch.zeros_like(mod.bias)
                        mod.bias.grad.add_((self._grad_scale * xi.sum(dim=0)).to(mod.bias.dtype))
                y_noisy = (y32 + xi).to(y_dtype) if self._add_noise else y
            self._row_counters[name] += N
            return y_noisy.reshape(*list(x_in.shape[:-1]) + [y_noisy.shape[-1]]) if reshape_back else y_noisy
        mod.forward = fwd
        self._orig[name] = orig


def load_conciseness_dataset(n_train, n_eval):
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache", "hf_datasets")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        ds = load_dataset("EdinburghNLP/xsum", "default", split=f"train[:{n_train + n_eval}]", cache_dir=cache_dir)
    except Exception as e1:
        try:
            ds = load_dataset("cnn_dailymail", "3.0.0", split=f"train[:{n_train + n_eval}]", cache_dir=cache_dir)
        except Exception as e2:
            print(f"Warning: Could not load xsum or cnn_dailymail. Using synthetic data.")
            return _synthetic_conciseness_data(n_train, n_eval)
    train_data, eval_data = [], []
    for i, ex in enumerate(ds):
        doc = ex.get("document", ex.get("article", ex.get("text", "")))
        summary = ex.get("summary", ex.get("highlights", ""))
        if isinstance(summary, list):
            summary = " ".join(summary)
        if not doc or not summary:
            continue
        prompt = f"Make the following text more concise:\n\n{doc[:1500]}\n\nConcise version:"
        target = f" {summary[:300]}"
        if len(train_data) < n_train:
            train_data.append((prompt, target))
        elif len(eval_data) < n_eval:
            eval_data.append((prompt, target))
        if len(train_data) >= n_train and len(eval_data) >= n_eval:
            break
    return train_data, eval_data


def _synthetic_conciseness_data(n_train, n_eval):
    examples = [
        ("The quick brown fox jumps over the lazy dog repeatedly.", "A fox jumps over a dog."),
        ("In 2024, many people work from home.", "People work from home."),
        ("The university was founded in 1850.", "The university was founded in 1850."),
        ("She went to the store to buy groceries.", "She bought groceries."),
        ("The meeting was delayed until 4pm.", "The meeting was delayed to 4pm."),
    ]
    expanded = (examples * ((n_train + n_eval) // len(examples) + 1))[:n_train + n_eval]
    fmt = lambda e: (f"Make the following text more concise:\n\n{e[0]}\n\nConcise version:", f" {e[1]}")
    return [fmt(e) for e in expanded[:n_train]], [fmt(e) for e in expanded[n_train:n_train + n_eval]]


def compute_loss(model, tokenizer, batch, device, label_ignore_id=-100):
    input_ids_list, labels_list = [], []
    for p, t in batch:
        prompt_ids = tokenizer.encode(p, add_special_tokens=True)
        target_ids = tokenizer.encode(t, add_special_tokens=False)
        full_ids = (prompt_ids + target_ids)[:args.max_length]
        lab = [label_ignore_id] * len(prompt_ids) + target_ids
        lab = lab[:args.max_length]
        input_ids_list.append(full_ids)
        labels_list.append(lab)
    pad_id = tokenizer.pad_token_id or 0
    max_len = max(len(x) for x in input_ids_list)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), label_ignore_id, dtype=torch.long)
    for i in range(len(batch)):
        L = len(input_ids_list[i])
        input_ids[i, :L] = torch.tensor(input_ids_list[i])
        attention_mask[i, :L] = 1
        labels[i, :L] = torch.tensor(labels_list[i])
    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
    return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss


def run_danp_step(model, tokenizer, batch, device, eta, sigma, base_seed, include, last_k):
    model.eval()
    for _, p in _named_linear_params(model, include, last_k):
        if p.grad is not None:
            p.grad.zero_()
    hook = DANPActivationHook(model, include=include, last_k=last_k, sigma=sigma, base_seed=base_seed)
    hook.attach(add_noise=False, track=False)
    with torch.no_grad():
        L_clean = compute_loss(model, tokenizer, batch, device).item()
    hook.detach()
    hook.attach(add_noise=True, track=False)
    with torch.no_grad():
        L_noisy = compute_loss(model, tokenizer, batch, device).item()
    norm_sq, N = hook.norm_sq + 1e-8, hook.total_N
    hook.detach()
    delta_L = L_noisy - L_clean
    scale = np.clip(eta * N * delta_L / norm_sq, -1e2, 1e2)
    hook.attach(add_noise=True, track=True, grad_scale=scale)
    _ = compute_loss(model, tokenizer, batch, device)
    hook.detach()
    with torch.no_grad():
        for _, p in _named_linear_params(model, include, last_k):
            if p.grad is not None:
                p.data.sub_(p.grad)
                p.grad = None
    return L_clean, L_noisy, delta_L


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if "HF_HOME" not in os.environ and "HUGGINGFACE_HUB_CACHE" not in os.environ:
        hf_cache = os.path.join(os.path.dirname(__file__), ".cache", "huggingface")
        os.makedirs(hf_cache, exist_ok=True)
        os.environ["HF_HOME"] = hf_cache
    print("Loading conciseness dataset...")
    train_data, eval_data = load_conciseness_dataset(args.n_train, args.n_eval)
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
    print(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.hf_cache_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.hf_cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        baseline = np.mean([compute_loss(model, tokenizer, eval_data[i:i+args.batch_size], device).item()
                           for i in range(0, len(eval_data), args.batch_size)])
    print(f"[BASELINE] Eval loss: {baseline:.4f}")
    train_losses, eval_losses_hist = [], []
    for epoch in range(args.epochs):
        model.train()
        indices = np.random.permutation(len(train_data))
        epoch_losses = []
        pbar = tqdm(range(0, len(train_data), args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, start in enumerate(pbar):
            batch = [train_data[i] for i in indices[start:start + args.batch_size]]
            L_clean, _, delta_L = run_danp_step(model, tokenizer, batch, device, args.eta, args.sigma,
                                                args.seed + epoch * 10000 + step, args.np_include, args.last_k)
            epoch_losses.append(L_clean)
            pbar.set_postfix({"loss": f"{L_clean:.4f}", "delta_L": f"{delta_L:.4f}"})
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)
        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                evals = [compute_loss(model, tokenizer, eval_data[i:i+args.batch_size], device).item()
                         for i in range(0, len(eval_data), args.batch_size)]
            eval_loss = np.mean(evals)
            eval_losses_hist.append(eval_loss)
            print(f"[Epoch {epoch+1}] Train: {train_loss:.4f}, Eval: {eval_loss:.4f}")
    print(f"\nDone. Final train loss: {train_losses[-1]:.4f}")
    if eval_losses_hist:
        print(f"Final eval loss: {eval_losses_hist[-1]:.4f}")
    save_dir = os.path.join(args.output_dir, "final_model")
    print(f"Saving to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Done.")


if __name__ == "__main__":
    main()
