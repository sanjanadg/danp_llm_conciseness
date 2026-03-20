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


  # Standard run
  # python3 danp_llm_conciseness.py --n_train 32 --n_eval 50 --epochs 10 --batch_size 8 --verbose --eta 1e-3 --sigma 1e-3 --alpha 1e-4


  # With verbose logging
  python3 danp_llm_conciseness.py --n_train 4 --n_eval 10 --epochs 10 --batch_size 2 --verbose --eta 1e-7 --sigma 1e-2 --alpha 0

  # Population size 4 (more stable gradient estimate)
  python3 danp_llm_conciseness.py --n_train 8 --n_eval 4 --epochs 10 --batch_size 2 --n_population 4 --eta 1e-7

  # Shorter sequences
  python danp_llm_conciseness.py --n_train 4 --n_eval 4 --epochs 1 --batch_size 2 --max_length 256

  # Copy task (echo input) – good for tiny models
  python3 danp_llm_conciseness.py --tiny_model trl --task copy --n_train 100 --n_eval 4 --epochs 5

  # Constant task (single-token target) – simplest
  python3 danp_llm_conciseness.py --tiny_model tiny_llama --task constant --n_train 8 --n_eval 4 --epochs 10

  # Conciseness (default)
  python3 danp_llm_conciseness.py --tiny_model tiny_llama --n_train 4 --n_eval 2 --epochs 2 --batch_size 2 --alpha 0

  # Quick DANP testing with tiny models:
  python3 danp_llm_conciseness.py --tiny_model trl --n_train 50 --n_eval 50 --epochs 3 --batch_size 4
  python3 danp_llm_conciseness.py --tiny_model tiny_gpt2 --n_train 50 --n_eval 50 --epochs 30 --batch_size 4 --verbose --eta 1e-8 --sigma 1e-2 --alpha 0 
  python3 danp_llm_conciseness.py --tiny_model gpt2 --n_train 50 --n_eval 50 --epochs 30 --batch_size 4 --verbose --eta 1e-8 --sigma 1e-2 --alpha 0 
  python3 danp_llm_conciseness.py --tiny_model tiny_llama --n_train 4 --n_eval 2 --epochs 2 --batch_size 2 --alpha 0

  # Simpler tasks (good for tiny models):
  python3 danp_llm_conciseness.py --tiny_model tiny_llama --task copy --n_train 8 --n_eval 4 --epochs 5
  python3 danp_llm_conciseness.py --tiny_model tiny_llama --task constant --n_train 8 --n_eval 4 --epochs 10

  # If delta_L≈0 (no learning): try larger sigma (e.g. --sigma 1e-2) so noise affects loss
  # If NaN: use smaller eta/alpha, or --max_dec_update 0.001 to stabilize decorrelation
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

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    plt = None

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

ETA = 1e-5
SIGMA = 1e-2
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
parser.add_argument('--max_dec_update', type=float, default=0.01, help='Clip decorrelation update to prevent R explosion/NaN')
parser.add_argument('--tiny_model', type=str, default=None,
                    help='Quick-test preset: trl (~2M), tiny_gpt2 (~4M), gpt2 (~124M), tiny_llama (~2K). Overrides --model_name and --max_length.')
parser.add_argument('--task', type=str, default='conciseness', choices=['copy', 'constant', 'conciseness'],
                    help='Task: copy (echo input), constant (fixed completion), conciseness (summarize).')
parser.add_argument('--no_plot', action='store_true', help='Disable saving loss plot')
parser.add_argument('--stable', action='store_true',
                    help='Use stable preset: tiny_gpt2 + copy task + max_length 64 (avoids NaN)')
args = parser.parse_args()

# Apply tiny_model preset
_TINY_PRESETS = {
    'trl': ('trl-internal-testing/tiny-LlamaForCausalLM-3', 128),
    'tiny_gpt2': ('sshleifer/tiny-gpt2', 128),
    'gpt2': ('gpt2', 64),
    'tiny_llama': ('ccmodular/tiny-random-LlamaForCausalLM', 32),
}
if args.tiny_model:
    if args.tiny_model not in _TINY_PRESETS:
        raise ValueError(f"--tiny_model must be one of {list(_TINY_PRESETS.keys())}, got {args.tiny_model!r}")
    args.model_name, args.max_length = _TINY_PRESETS[args.tiny_model]
if args.stable:
    args.model_name, args.max_length = _TINY_PRESETS['tiny_gpt2']
    args.task = 'copy'


# Architecture patterns: (regex for layer index, layer block name for last_k)
_ARCH_PATTERNS = [
    (r'\.layers\.(\d+)\.', 'layers'),   # Llama, Qwen, TinyLlama
    (r'\.h\.(\d+)\.', 'h'),              # GPT-2, distilgpt2
]

def _detect_layer_pattern(model):
    """Detect which architecture pattern the model uses. Returns (regex_str, total_blocks)."""
    for pattern, block_name in _ARCH_PATTERNS:
        names = [n for n, _ in model.named_modules() if re.search(pattern, n)]
        if names:
            idxs = [int(re.search(pattern, n).group(1)) for n in names]
            return pattern, max(idxs) + 1
    return None, 0


def _stable_uid(s: str) -> int:
    h = hashlib.blake2s(s.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(h[:4], 'little') & 0x7FFFFFFF


def _infer_total_blocks(model) -> int:
    _, total = _detect_layer_pattern(model)
    return total


def _get_layer_index(name: str, pattern: str):
    """Extract layer index from module name, or None if no match."""
    m = re.search(pattern, name)
    return int(m.group(1)) if m else None


def _is_target_linear(include: str, name: str, mod, last_k: int, total_blocks: int) -> bool:
    if not isinstance(mod, torch.nn.Linear):
        return False
    lname = name.lower()
    if last_k > 0 and total_blocks > 0:
        for pattern, _ in _ARCH_PATTERNS:
            idx = _get_layer_index(name, pattern)
            if idx is not None:
                if idx < total_blocks - last_k:
                    return False
                break
    if include == 'head':
        return lname.endswith('lm_head') or 'lm_head' in lname
    if include == 'attn':
        return any(k in lname for k in ('attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj', 'c_attn'))
    if include == 'mlp':
        return any(k in lname for k in ('mlp', 'ffn', 'gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2', 'c_fc', 'c_proj'))
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

    Llama/Qwen MLP: gate_proj and up_proj receive from o_proj. down_proj receives from gate*up.
    GPT-2 MLP: c_fc receives from residual. c_proj receives from c_fc (after GELU).
    """
    if "down_proj" in name:
        producer = name.replace("down_proj", "gate_proj")
        if producer in targets_by_name and targets_by_name[producer].out_features == in_features:
            return producer
    elif "gate_proj" in name or "up_proj" in name:
        producer = name.replace("mlp.gate_proj", "self_attn.o_proj").replace("mlp.up_proj", "self_attn.o_proj")
        if producer in targets_by_name and targets_by_name[producer].out_features == in_features:
            return producer
    elif "c_proj" in name:
        producer = name.replace("c_proj", "c_fc")
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
        self._pop_offset = 0

    def _seed_for_call(self, layer_uid: int, row_start: int) -> int:
        x = (self.base_seed ^ layer_uid) + 0x9e3779b9 + (row_start * 2654435761 & 0x7FFFFFFF) + (self._pop_offset * 1000000)
        return x & 0x7FFFFFFF

    def attach(self, add_noise: bool = True, capture: bool = False, pop_offset: int = 0):
        self._add_noise = add_noise
        self._capture = capture
        self._captured = {}
        self._pop_offset = pop_offset
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


def _prepare_batch(batch, tokenizer, device, max_length, label_ignore_id=-100, vocab_size=None):
    """Prepare input_ids, attention_mask, labels for a batch of (prompt, target) pairs.
    If vocab_size is provided, token IDs are clamped to [0, vocab_size-1] to avoid
    embedding IndexError when tokenizer vocab exceeds model embedding size (e.g. tiny Llama)."""
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
        inp = torch.tensor(input_ids_list[i], device=device)
        lab = torch.tensor(labels_list[i], device=device)
        if vocab_size is not None:
            inp = inp.clamp(0, vocab_size - 1)
            lab = torch.where(lab == label_ignore_id, lab, lab.clamp(0, vocab_size - 1))
        input_ids[i, :L] = inp
        attention_mask[i, :L] = 1
        labels[i, :L] = lab
    return input_ids, attention_mask, labels


def compute_loss(model, tokenizer, batch, device, label_ignore_id=-100):
    """Teacher-forcing cross-entropy loss on target tokens only. Uses float32 + logits clamp to avoid overflow/NaN."""
    input_ids, attention_mask, labels = _prepare_batch(
        batch, tokenizer, device, args.max_length, label_ignore_id,
        vocab_size=model.config.vocab_size,
    )
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits_f = logits.float().view(-1, model.config.vocab_size).clamp(-50.0, 50.0)
        loss = F.cross_entropy(
            logits_f, labels.view(-1), ignore_index=label_ignore_id,
        )
    return loss


def compute_danp_grad_estimate_per_sample(
    model, tokenizer, single_example, device, hook: DANPFullHook, eta, alpha,
    n_population=1, verbose=False, max_scale=1e2, max_update=1.0, max_dec_update=0.01,
):
    """
    Compute gradient estimate and decorrelation updates for a single sample.
    With n_population > 1: run multiple noisy forwards and average gradient estimates.
    Returns (weight_updates_dict, decorrelation_updates_dict, L_clean, delta_L_mean).
    """
    batch = [single_example]
    input_ids, attention_mask, labels = _prepare_batch(
        batch, tokenizer, device, args.max_length, label_ignore_id=-100,
        vocab_size=model.config.vocab_size,
    )

    # Pass 1: Clean forward (once)
    hook.attach(add_noise=False, capture=True)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits_f = logits.float().view(-1, model.config.vocab_size).clamp(-50.0, 50.0)
        L_clean = F.cross_entropy(
            logits_f, labels.view(-1), ignore_index=-100,
        ).item()
    if not np.isfinite(L_clean):
        hook.detach()
        return {}, {}, float('nan'), 0.0  # Skip corrupted sample
    captured_clean = {k: (v[0].clone(), v[1].clone(), None) for k, v in hook._captured.items()}
    hook.detach()

    # Pass 2: Noisy forward(s) - accumulate gradient estimates
    accumulated_grads = {}
    delta_L_sum = 0.0
    last_scale_raw, last_norm_sq, last_N = 0.0, 0.0, 0
    for pop_idx in range(n_population):
        hook.attach(add_noise=True, capture=True, pop_offset=pop_idx)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            logits_f = logits.float().view(-1, model.config.vocab_size).clamp(-50.0, 50.0)
            L_noisy = F.cross_entropy(
                logits_f, labels.view(-1), ignore_index=-100,
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
        if not np.isfinite(norm_sq):
            continue  # Skip population member with NaN activations

        delta_L = L_noisy - L_clean
        delta_L_clamped = np.clip(float(delta_L), -1e4, 1e4)
        if not np.isfinite(delta_L_clamped):
            delta_L_clamped = 0.0
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
            if not torch.isfinite(grad).all():
                continue
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
    # Clip updates and guard against NaN to prevent R explosion in deep LLMs
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
        dec_update = torch.clamp(dec_update, -max_dec_update, max_dec_update)
        if not (torch.isfinite(dec_update).all()):
            dec_update = torch.zeros_like(dec_update)
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
                        n_population=1, verbose=False, max_scale=1e2, max_update=1.0, max_dec_update=0.01):
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
            max_scale=max_scale, max_update=max_update, max_dec_update=max_dec_update,
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

    # Average and apply weight updates (skip if NaN to prevent corruption)
    with torch.no_grad():
        for key, acc in accumulated_weight_updates.items():
            upd = acc / batch_size
            if not torch.isfinite(upd).all():
                continue
            if key.endswith(".weight"):
                mod_path = key[:-7]
                mod = model.get_submodule(mod_path)
                mod.weight.sub_(upd.to(mod.weight.dtype))
            elif key.endswith(".bias"):
                mod_path = key[:-6]
                mod = model.get_submodule(mod_path)
                if mod.bias is not None:
                    mod.bias.sub_(upd.to(mod.bias.dtype))

    # Apply averaged decorrelation updates (with NaN guard)
    for name, acc in accumulated_decorrelation_updates.items():
        if acc is not None:
            upd = acc / batch_size
            if torch.isfinite(upd).all():
                hook.R[name].sub_(upd)
                if not torch.isfinite(hook.R[name]).all():
                    hook.R[name].copy_(torch.eye(hook.R[name].shape[0], device=hook.R[name].device, dtype=hook.R[name].dtype))
            # else: skip corrupted decorrelation update

    return total_loss / batch_size, total_delta_L / batch_size


def load_task_data(n_train, n_eval):
    """Load (train_data, eval_data) based on --task. Each item is (prompt, target)."""
    if args.task == 'copy':
        return _copy_task_data(n_train, n_eval)
    if args.task == 'constant':
        return _constant_task_data(n_train, n_eval)
    return load_conciseness_dataset(n_train, n_eval)


def _copy_task_data(n_train, n_eval):
    """Copy/echo task: Repeat the input. Short targets, easy for tiny models."""
    examples = [
        "hello", "world", "yes", "no", "cat", "dog", "run", "go",
        "one two", "a b c", "foo bar", "test", "ok", "hi", "bye",
        "red blue", "up down", "big small", "fast slow", "hot cold",
    ]
    n_total = n_train + n_eval
    expanded = (examples * ((n_total // len(examples)) + 1))[:n_total]
    train = [(f"Repeat: {e}", f" {e}") for e in expanded[:n_train]]
    eval_ = [(f"Repeat: {e}", f" {e}") for e in expanded[n_train:n_train + n_eval]]
    return train, eval_


def _constant_task_data(n_train, n_eval):
    """Constant completion: same prompt, same single-token target. Easiest task."""
    prompt = "The answer is:"
    target = " 4"  # single token (space + digit)
    train = [(prompt, target)] * n_train
    eval_ = [(prompt, target)] * n_eval
    return train, eval_


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

    print(f"Loading {args.task} dataset...")
    train_data, eval_data = load_task_data(args.n_train, args.n_eval)
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    hf_cache = args.hf_cache_dir or os.path.join(os.path.dirname(__file__), ".cache", "huggingface")
    os.makedirs(hf_cache, exist_ok=True)
    print(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=hf_cache,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=hf_cache)
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
    baseline_eval = np.nanmean(eval_losses)
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
            L_clean, delta_L =             run_danp_batch_step(
                model, tokenizer, batch, device, hook,
                eta=args.eta, alpha=args.alpha,
                n_population=args.n_population,
                verbose=verbose_this_step,
                max_scale=args.max_scale,
                max_update=args.max_update,
                max_dec_update=args.max_dec_update,
            )
            epoch_losses.append(L_clean)
            pbar.set_postfix({"loss": f"{L_clean:.4f}", "delta_L": f"{delta_L:.4f}"})

        train_loss = np.nanmean(epoch_losses)
        train_losses.append(train_loss)

        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                evals = []
                for i in range(0, len(eval_data), args.batch_size):
                    batch = eval_data[i:i + args.batch_size]
                    loss = compute_loss(model, tokenizer, batch, device).item()
                    evals.append(loss)
            eval_loss = np.nanmean(evals)
            eval_losses_hist.append(eval_loss)
            print(f"[Epoch {epoch+1}] Train loss: {train_loss:.4f}, Eval loss: {eval_loss:.4f}")

    print(f"\nTraining complete. Final train loss: {train_losses[-1]:.4f}")
    if eval_losses_hist:
        print(f"Final eval loss: {eval_losses_hist[-1]:.4f}")

    save_dir = os.path.join(args.output_dir, "final_model")
    print(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    if _HAS_MATPLOTLIB and not args.no_plot and train_losses:
        _plot_losses(train_losses, eval_losses_hist, args.eval_interval, args.output_dir, "conciseness")
    elif not args.no_plot and train_losses and not _HAS_MATPLOTLIB:
        print("Tip: Install matplotlib (pip install matplotlib) to save loss plots.")

    print("Done.")


def _plot_losses(train_losses, eval_losses_hist, eval_interval, output_dir, experiment_name):
    """Save train/eval loss plot to output_dir."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, 'b-o', markersize=4, label='Train loss')
    if eval_losses_hist:
        eval_epochs = [eval_interval * (i + 1) for i in range(len(eval_losses_hist))]
        ax.plot(eval_epochs, eval_losses_hist, 'r-s', markersize=4, label='Eval loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'DANP {experiment_name} - Train & Eval Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, 'loss_plot.png')
    fig.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved loss plot to {plot_path}")


if __name__ == "__main__":
    main()
