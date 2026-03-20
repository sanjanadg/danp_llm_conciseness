#!/usr/bin/env python3
"""
Full DANP (Decorrelated activity based Node Perturbation) for finetuning Qwen 0.5B on conciseness task.

Implements Algorithm 1 with:
- Decorrelation: R matrices per layer, x* = R @ x in forward, R_l -= α(x*_l(x*_l)^T - diag((x*_l)²)) R_l
- Weight update: W_l -= η N δL (ã_l - a_l) / ||δa||² (x*_{l-1})^T
- Batched training: per-sample gradient/R accumulation, then average and apply (like train_danp_batch)

Usage:
  python danp_llm_conciseness.py --n_train 32 --n_eval 50 --epochs 5 --batch_size 2
  python danp_llm_conciseness.py --model_name Qwen/Qwen2.5-0.5B-Instruct --output_dir ./out_danp_full

  # Minimal run
  python danp_llm_conciseness.py --n_train 4 --n_eval 4 --epochs 1 --batch_size 2

  # With verbose logging
  python danp_llm_conciseness.py --n_train 4 --n_eval 4 --epochs 4 --batch_size 2 --verbose

  # Population size 4 (more stable gradient estimate)
  python danp_llm_conciseness.py --n_train 8 --n_eval 4 --epochs 2 --batch_size 2 --n_population 4

  # Shorter sequences
  python danp_llm_conciseness.py --n_train 4 --n_eval 4 --epochs 1 --batch_size 2 --max_length 256
"""

import torch
import torch.nn.functional as F
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

ETA = 1e-3
SIGMA = 1e-3
ALPHA = 1e-4

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='./out_danp_conciseness_full')
parser.add_argument('--n_train', type=int, default=64)
parser.add_argument('--n_eval', type=int, default=50)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--eta', type=float, default=ETA)
parser.add_argument('--sigma', type=float, default=SIGMA)
parser.add_argument('--alpha', type=float, default=ALPHA)
parser.add_argument('--np_include', type=str, default='mlp', choices=['head', 'attn', 'mlp', 'all'])
parser.add_argument('--last_k', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--eval_interval', type=int, default=1)
parser.add_argument('--verbose', action='store_true', help='Print gradient/update stats for debugging')
parser.add_argument('--n_population', type=int, default=1,
                    help='Number of noisy forward passes per sample; average gradient estimate (reduces variance)')
parser.add_argument('--max_scale', type=float, default=1e2, help='Clip scale factor to prevent explosion')
parser.add_argument('--max_update', type=float, default=1.0, help='Clip per-layer weight update magnitude')
args = parser.parse_args()


def _stable_uid(s: str) -> int:
    h = hashlib.blake2s(s.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(h[:4], 'little') & 0x7FFFFFFF


def _infer_total_blocks(model) -> int:
    names = [n for n, _ in model.named_modules() if re.search(r'\.layers\.(\d+)\.', n)]
    if not names:
        return 0
    idxs = [int(re.search(r'\.layers\.(\d+)\.', n).group(1)) for n in names]
    return max(idxs) + 1


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
        return any(k in lname for k in ('attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj'))
    if include == 'mlp':
        return any(k in lname for k in ('mlp', 'ffn', 'gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2'))
    return True


def _get_target_linears_ordered(model, include, last_k):
    """Return ordered list of (name, mod) for target Linears."""
    total_blocks = _infer_total_blocks(model)
    targets = []
    for name, mod in model.named_modules():
        if _is_target_linear(include, name, mod, last_k, total_blocks):
            targets.append((name, mod))
    return targets


def _get_R_in_producer(name, in_features, targets_by_name):
    """
    For a Linear with given name and in_features, return the name of the layer whose
    R_out should be used as R_in (i.e. the layer that produces this input).
    Returns None if input comes from outside our hooked set.
    Qwen2 MLP: gate_proj and up_proj receive same 896-dim input (from o_proj/residual).
    down_proj receives 4864-dim from gate*up.
    """
    if "down_proj" in name:
        # down_proj receives from gate_proj output (or up_proj, same dim)
        producer = name.replace("down_proj", "gate_proj")
        if producer in targets_by_name and targets_by_name[producer].out_features == in_features:
            return producer
    elif "gate_proj" in name or "up_proj" in name:
        # gate_proj and up_proj receive from o_proj (896) - may not be in our targets
        producer = name.replace("mlp.gate_proj", "self_attn.o_proj").replace("mlp.up_proj", "self_attn.o_proj")
        if producer in targets_by_name and targets_by_name[producer].out_features == in_features:
            return producer
    return None


def _named_linear_params(model, include, last_k):
    total_blocks = _infer_total_blocks(model)
    for name, mod in model.named_modules():
        if _is_target_linear(include, name, mod, last_k, total_blocks):
            if getattr(mod, "weight", None) is not None:
                yield name + ".weight", mod.weight
            if getattr(mod, "bias", None) is not None:
                yield name + ".bias", mod.bias


class DANPFullHook:
    """
    Full DANP hook with decorrelation (Algorithm 1).
    - R matrices: one per Linear, R_out decorrelates output (shape out_features x out_features)
    - Forward: input -> R_in @ x (R_in from prev layer), output -> R_out @ (W @ x_star) [+ noise]
    - Weight update: grad = outer(delta_a, x_star), W -= eta * N * delta_L * grad / norm_sq
    - Decorrelation: R_out -= alpha * (cov - diag) @ R_out, cov = outer(x_star, x_star), x_star = R_out @ y
    """
    def __init__(self, model, include, last_k, sigma, alpha, base_seed):
        self.model = model
        self.include = include
        self.last_k = last_k
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.base_seed = int(base_seed)
        self._orig = {}
        self._targets = _get_target_linears_ordered(model, include, last_k)
        self._layer_uids = {name: _stable_uid(name) for name, _ in self._targets}
        self._row_counters = {name: 0 for name, _ in self._targets}

        # R matrices: one per Linear, decorrelates output. R[l] has shape (out_features, out_features)
        self.R = {}
        self._R_in = {}
        targets_by_name = {name: mod for name, mod in self._targets}
        for name, mod in self._targets:
            out_dim = mod.out_features
            in_dim = mod.in_features
            R_out = torch.eye(out_dim, dtype=torch.float32, device=mod.weight.device)
            self.R[name] = R_out
            producer = _get_R_in_producer(name, in_dim, targets_by_name)
            self._R_in[name] = self.R[producer] if producer else None

        self._add_noise = True
        self._capture = False
        self._captured = {}

    def _seed_for_call(self, layer_uid: int, row_start: int) -> int:
        x = (self.base_seed ^ layer_uid) + 0x9e3779b9 + (row_start * 2654435761 & 0x7FFFFFFF)
        return x & 0x7FFFFFFF

    def attach(self, add_noise: bool = True, capture: bool = False):
        self._add_noise = add_noise
        self._capture = capture
        self._captured = {}
        for name in self._row_counters:
            self._row_counters[name] = 0
        for name, mod in self._targets:
            self._wrap(name, mod)

    def detach(self):
        for name, mod in self._targets:
            if name in self._orig:
                mod.forward = self._orig[name]
        self._orig.clear()

    def _wrap(self, name, mod: torch.nn.Linear):
        orig = mod.forward
        layer_uid = self._layer_uids[name]
        R_out = self.R[name]
        R_in = self._R_in[name]

        def fwd(x):
            x_in = x
            if x_in.dim() > 2:
                N = int(np.prod(x_in.shape[:-1]))
                C = x_in.shape[-1]
                x2 = x_in.reshape(N, C).float()
                reshape_back = True
            else:
                x2 = x_in.float()
                N, C = x2.shape
                reshape_back = False

            # x_star = R_in @ x (Algorithm 1: input to layer is decorrelated)
            if R_in is not None:
                x_star = x2 @ R_in.T.to(x2.dtype)
            else:
                x_star = x2

            with torch.no_grad():
                y = orig(x_star.to(mod.weight.dtype))
                y_dtype = y.dtype
                y32 = y.float()
                out_dim = y32.shape[-1]
                dev = y32.device

                start = self._row_counters[name]
                g = torch.Generator(device=dev)
                g.manual_seed(self._seed_for_call(layer_uid, start))
                xi = torch.randn((N, out_dim), generator=g, device=dev, dtype=torch.float32) * self.sigma

                if self._add_noise:
                    y_noisy = y32 + xi
                else:
                    y_noisy = y32

                if self._capture:
                    self._captured[name] = (
                        x_star.detach().clone(),
                        y32.detach().clone(),
                        y_noisy.detach().clone() if self._add_noise else None,
                    )

                y_out = (y_noisy @ R_out.T.to(y_noisy.dtype)).to(y_dtype)

            self._row_counters[name] += N
            if reshape_back:
                new_shape = list(x_in.shape[:-1]) + [y_out.shape[-1]]
                y_out = y_out.reshape(*new_shape)
            return y_out

        mod.forward = fwd
        self._orig[name] = orig


def _prepare_batch(batch, tokenizer, device, max_length, label_ignore_id=-100):
    """Prepare input_ids, attention_mask, labels for a batch of (prompt, target) pairs."""
    input_ids_list = []
    labels_list = []
    for p, t in batch:
        prompt_ids = tokenizer.encode(p, add_special_tokens=True)
        target_ids = tokenizer.encode(t, add_special_tokens=False)
        full_ids = (prompt_ids + target_ids)[:max_length]
        lab = [label_ignore_id] * len(prompt_ids) + target_ids
        lab = lab[:max_length]
        input_ids_list.append(full_ids)
        labels_list.append(lab)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_len = max(len(x) for x in input_ids_list)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
    labels = torch.full((len(batch), max_len), label_ignore_id, dtype=torch.long, device=device)
    for i in range(len(batch)):
        L = len(input_ids_list[i])
        input_ids[i, :L] = torch.tensor(input_ids_list[i], device=device)
        attention_mask[i, :L] = 1
        labels[i, :L] = torch.tensor(labels_list[i], device=device)
    return input_ids, attention_mask, labels


def compute_loss(model, tokenizer, batch, device, label_ignore_id=-100):
    """Teacher-forcing cross-entropy loss on target tokens only."""
    input_ids, attention_mask, labels = _prepare_batch(
        batch, tokenizer, device, args.max_length, label_ignore_id
    )
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss


def compute_danp_grad_estimate_per_sample(
    model, tokenizer, single_example, device, hook: DANPFullHook, eta, alpha,
    n_population=1, verbose=False, max_scale=1e2, max_update=1.0,
):
    """
    Compute gradient estimate and decorrelation updates for a single sample.
    With n_population > 1: run multiple noisy forwards and average gradient estimates.
    Returns (weight_updates_dict, decorrelation_updates_dict, L_clean, delta_L_mean).
    """
    batch = [single_example]
    input_ids, attention_mask, labels = _prepare_batch(
        batch, tokenizer, device, args.max_length, label_ignore_id=-100
    )

    # Pass 1: Clean forward (once)
    hook.attach(add_noise=False, capture=True)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        L_clean = F.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        ).item()
    captured_clean = {k: (v[0].clone(), v[1].clone(), None) for k, v in hook._captured.items()}
    hook.detach()

    # Pass 2: Noisy forward(s) - accumulate gradient estimates
    accumulated_grads = {}
    delta_L_sum = 0.0
    last_scale_raw, last_norm_sq, last_N = 0.0, 0.0, 0
    for pop_idx in range(n_population):
        hook.attach(add_noise=True, capture=True)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            L_noisy = F.cross_entropy(
                logits.view(-1, model.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            ).item()
        captured_noisy = hook._captured.copy()
        hook.detach()

        delta_L_sum += L_noisy - L_clean

        delta_a_all = []
        for name, _ in hook._targets:
            if name not in captured_clean or name not in captured_noisy:
                continue
            _, y_clean, _ = captured_clean[name]
            _, _, y_noisy = captured_noisy[name]
            if y_noisy is not None:
                delta_a_all.append((y_noisy - y_clean).flatten())

        if not delta_a_all:
            continue

        delta_a_concat = torch.cat(delta_a_all)
        norm_sq = (delta_a_concat ** 2).sum().item() + 1e-8
        N = delta_a_concat.numel()

        delta_L = L_noisy - L_clean
        delta_L_clamped = np.clip(delta_L, -1e4, 1e4) # should clamp here?
        scale_raw = eta * N * delta_L_clamped / norm_sq
        scale = np.clip(scale_raw, -max_scale, max_scale)
        last_scale_raw, last_norm_sq, last_N = scale_raw, norm_sq, N

        for name, mod in hook._targets:
            if name not in captured_clean or name not in captured_noisy:
                continue
            x_star, y_clean, _ = captured_clean[name]
            _, _, y_noisy = captured_noisy[name]
            if y_noisy is None:
                continue
            delta_a = y_noisy - y_clean
            if x_star.dim() > 2:
                x_star = x_star.reshape(-1, x_star.shape[-1])
            if delta_a.dim() > 2:
                delta_a = delta_a.reshape(-1, delta_a.shape[-1])
            grad = delta_a.T @ x_star
            update = scale * grad
            update = torch.clamp(update, -max_update, max_update)
            key_w = name + ".weight"
            if key_w not in accumulated_grads:
                accumulated_grads[key_w] = torch.zeros_like(grad, device=grad.device)
            accumulated_grads[key_w].add_(update)
            if mod.bias is not None:
                bias_upd = scale * delta_a.sum(dim=0)
                bias_upd = torch.clamp(bias_upd, -max_update, max_update)
                key_b = name + ".bias"
                if key_b not in accumulated_grads:
                    accumulated_grads[key_b] = torch.zeros_like(mod.bias, device=mod.bias.device)
                accumulated_grads[key_b].add_(bias_upd)

    if not accumulated_grads:
        return {}, {}, L_clean, delta_L_sum / max(1, n_population)

    delta_L_mean = delta_L_sum / n_population

    # Average weight updates over population
    weight_updates = {}
    for key, acc in accumulated_grads.items():
        acc_mean = acc / n_population
        if key.endswith(".weight"):
            mod = model.get_submodule(key[:-7])
            weight_updates[key] = acc_mean.to(mod.weight.dtype)
        else:
            mod = model.get_submodule(key[:-6])
            if mod.bias is not None:
                weight_updates[key] = acc_mean.to(mod.bias.dtype)
    
    # Decorrelation: R -= alpha * (cov - diag) @ R (uses clean activations only)
    decorrelation_updates = {}
    for name, _ in hook._targets:
        if name not in captured_clean:
            continue
        _, y_clean, _ = captured_clean[name]
        R_out = hook.R[name]
        x_star_out = y_clean @ R_out.T
        if x_star_out.dim() > 2:
            x_star_out = x_star_out.reshape(-1, x_star_out.shape[-1])
        if x_star_out.shape[0] > 1:
            cov = x_star_out.T @ x_star_out / x_star_out.shape[0]
        else:
            cov = torch.outer(x_star_out.squeeze(0), x_star_out.squeeze(0))
        diag = torch.diag((x_star_out ** 2).mean(dim=0))
        dec_update = alpha * (cov - diag) @ R_out
        decorrelation_updates[name] = dec_update

    if verbose:
        _print_grad_stats(weight_updates, decorrelation_updates, L_clean, delta_L_mean, last_scale_raw, last_norm_sq, last_N)

    return weight_updates, decorrelation_updates, L_clean, delta_L_mean


def _print_grad_stats(weight_updates, decorrelation_updates, L_clean, delta_L, scale_raw, norm_sq, N):
    """Print gradient/update stats for debugging."""
    print(f"\n[DANP stats] L_clean={L_clean:.4f} delta_L={delta_L:.6f} scale_raw={scale_raw:.6e} norm_sq={norm_sq:.6e} N={N}")
    for key, upd in list(weight_updates.items())[:5]:  # first 5 layers
        u = upd.detach().float() if upd.is_cuda else upd.float()
        print(f"  {key}: update max={u.abs().max().item():.6e} mean={u.abs().mean().item():.6e} std={u.std().item():.6e}")
    if len(weight_updates) > 5:
        print(f"  ... ({len(weight_updates)} layers total)")


def run_danp_batch_step(model, tokenizer, batch, device, hook, eta, alpha,
                        n_population=1, verbose=False, max_scale=1e2, max_update=1.0):
    """
    One DANP batch step: per-sample gradient/R accumulation, then average and apply.
    Like train_danp_batch in sythentic_data.py.
    """
    batch_size = len(batch)
    accumulated_weight_updates = {}
    accumulated_decorrelation_updates = {name: None for name in hook.R}

    total_loss = 0.0
    total_delta_L = 0.0

    for i in range(batch_size):
        weight_updates, decorrelation_updates, L_clean, delta_L = compute_danp_grad_estimate_per_sample(
            model, tokenizer, batch[i], device, hook, eta, alpha,
            n_population=n_population, verbose=verbose and i == 0,
            max_scale=max_scale, max_update=max_update,
        )
        total_loss += L_clean
        total_delta_L += delta_L

        for key, upd in weight_updates.items():
            if key not in accumulated_weight_updates:
                accumulated_weight_updates[key] = torch.zeros_like(upd)
            accumulated_weight_updates[key] += upd

        for name, upd in decorrelation_updates.items():
            if accumulated_decorrelation_updates[name] is None:
                accumulated_decorrelation_updates[name] = torch.zeros_like(upd)
            accumulated_decorrelation_updates[name] += upd

    # Average and apply weight updates
    with torch.no_grad():
        for key, acc in accumulated_weight_updates.items():
            if key.endswith(".weight"):
                mod_path = key[:-7]
                mod = model.get_submodule(mod_path)
                mod.weight.sub_(acc.to(mod.weight.dtype) / batch_size)
            elif key.endswith(".bias"):
                mod_path = key[:-6]
                mod = model.get_submodule(mod_path)
                if mod.bias is not None:
                    mod.bias.sub_(acc.to(mod.bias.dtype) / batch_size)

    # Apply averaged decorrelation updates
    for name, acc in accumulated_decorrelation_updates.items():
        if acc is not None:
            hook.R[name].sub_(acc / batch_size)

    return total_loss / batch_size, total_delta_L / batch_size


def load_conciseness_dataset(n_train, n_eval):
    """Load conciseness/summarization dataset. Uses xsum for short summaries."""
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache", "hf_datasets")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        ds = load_dataset("EdinburghNLP/xsum", "default", split=f"train[:{n_train + n_eval}]", cache_dir=cache_dir)
    except Exception as e1:
        try:
            ds = load_dataset("cnn_dailymail", "3.0.0", split=f"train[:{n_train + n_eval}]", cache_dir=cache_dir)
        except Exception as e2:
            print(f"Warning: Could not load xsum ({e1}) or cnn_dailymail ({e2}). Using synthetic data.")
            return _synthetic_conciseness_data(n_train, n_eval)
    train_data = []
    eval_data = []
    for i, ex in enumerate(ds):
        doc = ex.get("document", ex.get("article", ex.get("text", "")))
        summary = ex.get("summary", ex.get("highlights", ""))
        if isinstance(summary, list):
            summary = " ".join(summary)
        if not doc or not summary:
            continue
        prompt = f"Make the following text more concise while preserving the key information:\n\n{doc[:1500]}\n\nConcise version:"
        target = f" {summary[:300]}"
        if len(train_data) < n_train:
            train_data.append((prompt, target))
        elif len(eval_data) < n_eval:
            eval_data.append((prompt, target))
        if len(train_data) >= n_train and len(eval_data) >= n_eval:
            break
    return train_data, eval_data


def _synthetic_conciseness_data(n_train, n_eval):
    """Fallback synthetic conciseness examples when dataset download fails."""
    examples = [
        ("The quick brown fox jumps over the lazy dog repeatedly and then does it again.",
         "A fox jumps over a dog."),
        ("In the year 2024, many people have been working from home for several years now.",
         "People work from home."),
        ("The university was founded in the year 1850 and has since grown to become one of the largest.",
         "The university was founded in 1850."),
        ("She went to the store to buy some groceries and then she came back home.",
         "She bought groceries."),
        ("The meeting was scheduled for 3pm but it got delayed until 4pm.",
         "The meeting was delayed to 4pm."),
    ]
    n_total = n_train + n_eval
    expanded = (examples * ((n_total // len(examples)) + 1))[:n_total]
    train_data = [(f"Make the following text more concise:\n\n{e[0]}\n\nConcise version:",
                   f" {e[1]}") for e in expanded[:n_train]]
    eval_data = [(f"Make the following text more concise:\n\n{e[0]}\n\nConcise version:",
                  f" {e[1]}") for e in expanded[n_train:n_train + n_eval]]
    return train_data, eval_data


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
        args.model_name,
        cache_dir=args.hf_cache_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.hf_cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Baseline eval
    model.eval()
    with torch.no_grad():
        eval_losses = []
        for i in range(0, len(eval_data), args.batch_size):
            batch = eval_data[i:i + args.batch_size]
            loss = compute_loss(model, tokenizer, batch, device).item()
            eval_losses.append(loss)
    baseline_eval = np.mean(eval_losses)
    print(f"[BASELINE] Eval loss: {baseline_eval:.4f}")

    train_losses = []
    eval_losses_hist = []

    for epoch in range(args.epochs):
        model.train()
        indices = np.random.permutation(len(train_data))
        epoch_losses = []

        hook = DANPFullHook(
            model, args.np_include, args.last_k,
            sigma=args.sigma, alpha=args.alpha,
            base_seed=args.seed + epoch * 10000,
        )
        for name in hook.R:
            hook.R[name] = hook.R[name].to(device)

        pbar = tqdm(range(0, len(train_data), args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, start in enumerate(pbar):
            batch_idx = indices[start:start + args.batch_size]
            batch = [train_data[i] for i in batch_idx]
            base_seed = args.seed + epoch * 10000 + step
            hook.base_seed = base_seed

            verbose_this_step = args.verbose and step == 0
            L_clean, delta_L = run_danp_batch_step(
                model, tokenizer, batch, device, hook,
                eta=args.eta, alpha=args.alpha,
                n_population=args.n_population,
                verbose=verbose_this_step,
                max_scale=args.max_scale,
                max_update=args.max_update,
            )
            epoch_losses.append(L_clean)
            pbar.set_postfix({"loss": f"{L_clean:.4f}", "delta_L": f"{delta_L:.4f}"})

        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                evals = []
                for i in range(0, len(eval_data), args.batch_size):
                    batch = eval_data[i:i + args.batch_size]
                    loss = compute_loss(model, tokenizer, batch, device).item()
                    evals.append(loss)
            eval_loss = np.mean(evals)
            eval_losses_hist.append(eval_loss)
            print(f"[Epoch {epoch+1}] Train loss: {train_loss:.4f}, Eval loss: {eval_loss:.4f}")

    print(f"\nTraining complete. Final train loss: {train_losses[-1]:.4f}")
    if eval_losses_hist:
        print(f"Final eval loss: {eval_losses_hist[-1]:.4f}")

    save_dir = os.path.join(args.output_dir, "final_model")
    print(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Done.")


if __name__ == "__main__":
    main()
