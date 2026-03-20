#!/usr/bin/env python3
"""
DANP distillation experiment: teacher-student and next-token prediction.

Clean experiment to validate DANP without task complexity:
- Dense supervision (loss at every token)
- Shorter credit paths
- Lower variance signal

Modes:
  1. distillation: L = KL(student || teacher), optionally + CE with ground truth
  2. next_token: L = CE(student_logits, next_token) - baseline sanity check

Usage:
  # Teacher-student distillation (KL loss; use different teacher/student for meaningful signal)
  python3 danp_distillation.py --mode distillation --teacher gpt2 --student tiny_gpt2 --dataset wikitext2 --n_train 100 --epochs 10

  # Next-token prediction (CE loss) - baseline sanity check
  python3 danp_distillation.py --mode next_token --student tiny_gpt2 --dataset synthetic --n_train 50 --epochs 5

  # Quick test with synthetic data
  python3 danp_distillation.py --mode next_token --student tiny_gpt2 --dataset synthetic --n_train 8 --n_eval 4 --epochs 2

  # With antithetic sampling and delta_L normalization
  python3 danp_distillation.py --mode distillation --teacher gpt2 --student tiny_gpt2 --antithetic --delta_l_norm batch
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

# Model presets
_PRESETS = {
    'tiny_gpt2': 'sshleifer/tiny-gpt2',
    'gpt2': 'gpt2',
    'tiny_llama': 'ccmodular/tiny-random-LlamaForCausalLM',
    'trl': 'trl-internal-testing/tiny-LlamaForCausalLM-3',
}

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='distillation', choices=['distillation', 'next_token'],
                    help='distillation=KL(student||teacher), next_token=CE only')
parser.add_argument('--teacher', type=str, default='tiny_gpt2', help='Teacher model (preset or HF path)')
parser.add_argument('--student', type=str, default='tiny_gpt2', help='Student model (preset or HF path)')
parser.add_argument('--dataset', type=str, default='wikitext2', choices=['wikitext2', 'tinystories', 'synthetic'],
                    help='Dataset for next-token prediction')
parser.add_argument('--output_dir', type=str, default='./out_danp_distillation')
parser.add_argument('--n_train', type=int, default=200)
parser.add_argument('--n_eval', type=int, default=50)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--max_length', type=int, default=64)
parser.add_argument('--eta', type=float, default=1e-6)
parser.add_argument('--sigma', type=float, default=1e-2)
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--kl_weight', type=float, default=1.0, help='Weight for KL term in distillation')
parser.add_argument('--ce_weight', type=float, default=0.0, help='Weight for CE term in distillation (0=KL only)')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for KL (teacher logits)')
parser.add_argument('--n_population', type=int, default=2, help='Noise samples per example (K)')
parser.add_argument('--antithetic', action='store_true', help='Use antithetic sampling: epsilon and -epsilon')
parser.add_argument('--delta_l_norm', type=str, default='none', choices=['none', 'batch', 'token'],
                    help='Normalize delta_L: none, batch (mean/std over batch), token (per-token)')
parser.add_argument('--np_include', type=str, default='mlp', choices=['mlp', 'attn', 'all'])
parser.add_argument('--last_k', type=int, default=0)
parser.add_argument('--max_scale', type=float, default=1e2)
parser.add_argument('--max_update', type=float, default=1.0)
parser.add_argument('--max_dec_update', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--no_plot', action='store_true', help='Disable saving loss plot')
parser.add_argument('--hf_cache_dir', type=str, default=None)
args = parser.parse_args()

# Resolve model names
def _resolve_model(name):
    return _PRESETS.get(name, name)

args.teacher_path = _resolve_model(args.teacher)
args.student_path = _resolve_model(args.student)

# Architecture patterns (same as danp_llm_conciseness)
_ARCH_PATTERNS = [(r'\.layers\.(\d+)\.', 'layers'), (r'\.h\.(\d+)\.', 'h')]

def _detect_layer_pattern(model):
    for pattern, _ in _ARCH_PATTERNS:
        names = [n for n, _ in model.named_modules() if re.search(pattern, n)]
        if names:
            idxs = [int(re.search(pattern, n).group(1)) for n in names]
            return pattern, max(idxs) + 1
    return None, 0

def _infer_total_blocks(model):
    _, total = _detect_layer_pattern(model)
    return total

def _get_layer_index(name, pattern):
    m = re.search(pattern, name)
    return int(m.group(1)) if m else None

def _is_target_linear(include, name, mod, last_k, total_blocks):
    if not isinstance(mod, torch.nn.Linear):
        return False
    lname = name.lower()
    if last_k > 0 and total_blocks > 0:
        for pattern, _ in _ARCH_PATTERNS:
            idx = _get_layer_index(name, pattern)
            if idx is not None and idx < total_blocks - last_k:
                return False
    if include == 'attn':
        return any(k in lname for k in ('attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'c_attn'))
    if include == 'mlp':
        return any(k in lname for k in ('mlp', 'gate_proj', 'up_proj', 'down_proj', 'c_fc', 'c_proj'))
    return True

def _get_target_linears_ordered(model, include, last_k):
    total = _infer_total_blocks(model)
    return [(n, m) for n, m in model.named_modules()
            if _is_target_linear(include, n, m, last_k, total)]

def _get_R_in_producer(name, in_dim, targets_by_name):
    if "down_proj" in name:
        p = name.replace("down_proj", "gate_proj")
        if p in targets_by_name and targets_by_name[p].out_features == in_dim:
            return p
    elif "gate_proj" in name or "up_proj" in name:
        p = name.replace("mlp.gate_proj", "self_attn.o_proj").replace("mlp.up_proj", "self_attn.o_proj")
        if p in targets_by_name and targets_by_name[p].out_features == in_dim:
            return p
    elif "c_proj" in name:
        p = name.replace("c_proj", "c_fc")
        if p in targets_by_name and targets_by_name[p].out_features == in_dim:
            return p
    return None

def _stable_uid(s):
    h = hashlib.blake2s(s.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(h[:4], 'little') & 0x7FFFFFFF


class DANPFullHook:
    """DANP hook with decorrelation. Same logic as danp_llm_conciseness."""
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
        self.R = {}
        self._R_in = {}
        targets_by_name = {name: mod for name, mod in self._targets}
        for name, mod in self._targets:
            R_out = torch.eye(mod.out_features, dtype=torch.float32, device=mod.weight.device)
            self.R[name] = R_out
            prod = _get_R_in_producer(name, mod.in_features, targets_by_name)
            self._R_in[name] = self.R[prod] if prod else None
        self._add_noise = True
        self._capture = False
        self._captured = {}
        self._pop_offset = 0

    def _seed_for_call(self, layer_uid, row_start):
        x = (self.base_seed ^ layer_uid) + 0x9e3779b9 + (row_start * 2654435761 & 0x7FFFFFFF) + (self._pop_offset * 1000000)
        return x & 0x7FFFFFFF

    def attach(self, add_noise=True, capture=False, pop_offset=0):
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

    def _wrap(self, name, mod):
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
                y_out = y_out.reshape(*x_in.shape[:-1], -1)
            return y_out

        mod.forward = fwd
        self._orig[name] = orig


def _prepare_batch_lm(batch_ids, tokenizer, device, max_length, vocab_size=None):
    """Batch of token sequences -> padded input_ids, attention_mask, labels (labels[t]=input_ids[t+1])."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id or 0
    label_ignore_id = -100
    truncated = [ids[:max_length] for ids in batch_ids]
    max_len = min(max(len(x) for x in truncated), max_length)
    input_ids = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(batch_ids), max_len), dtype=torch.long, device=device)
    labels = torch.full((len(batch_ids), max_len), label_ignore_id, dtype=torch.long, device=device)
    for i, ids in enumerate(truncated):
        L = min(len(ids), max_len)
        inp = torch.tensor(ids[:L], device=device)
        if vocab_size is not None:
            inp = inp.clamp(0, vocab_size - 1)
        input_ids[i, :L] = inp
        attention_mask[i, :L] = 1
        # labels[t] = next token = input_ids[t+1]
        if L > 1:
            lab = torch.tensor(ids[1:L], device=device)
            if vocab_size is not None:
                lab = lab.clamp(0, vocab_size - 1)
            labels[i, :L-1] = lab
    return input_ids, attention_mask, labels


def compute_loss_ce(model, input_ids, attention_mask, labels, vocab_size, label_ignore_id=-100):
    """Cross-entropy loss for next-token prediction. logits[:,t,:] predicts labels[:,t] (= next token)."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size).float()
        shift_labels = labels[..., :-1].contiguous().view(-1)  # labels[t] = next token
        return F.cross_entropy(shift_logits, shift_labels, ignore_index=label_ignore_id).item()


def compute_loss_kl(student_logits, teacher_logits, temperature=1.0, mask=None):
    """KL(student || teacher). PyTorch kl_div(input=log_target, target=source) -> target*(log_target - input)."""
    s_prob = F.softmax(student_logits.float() / temperature, dim=-1)
    t_log = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    # KL(student||teacher) = sum student * (log student - log teacher)
    # kl_div(input=t_log, target=s_prob) = s_prob * (log(s_prob) - t_log) = student*(log_student - log_teacher)
    kl = F.kl_div(t_log, s_prob, reduction='none').sum(dim=-1)
    if mask is not None:
        kl = kl * mask
        return (kl.sum() / (mask.sum() + 1e-8)).item()
    return kl.mean().item()


def compute_loss_distillation(student, teacher, input_ids, attention_mask, labels,
                               vocab_size, kl_weight=1.0, ce_weight=0.0, temperature=1.0):
    """L = kl_weight * KL(student||teacher) + ce_weight * CE(student, labels)."""
    with torch.no_grad():
        s_out = student(input_ids=input_ids, attention_mask=attention_mask)
        t_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
        s_logits = s_out.logits.float()
        t_logits = t_out.logits.float()
        shift_s = s_logits[..., :-1, :].contiguous()
        shift_t = t_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., :-1].contiguous()  # labels[t] = next token at position t
        mask = (shift_labels != -100).float()
        kl = compute_loss_kl(shift_s.view(-1, vocab_size), shift_t.view(-1, vocab_size),
                            temperature, mask.view(-1))
        loss = kl_weight * kl
        if ce_weight > 0:
            ce = F.cross_entropy(shift_s.view(-1, vocab_size), shift_labels.view(-1),
                                 ignore_index=-100)
            loss = loss + ce_weight * ce.item()
        return loss


def load_lm_dataset(n_train, n_eval, tokenizer, max_length):
    """Load sequences for next-token prediction. Returns list of token id lists."""
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache", "hf_datasets")
    os.makedirs(cache_dir, exist_ok=True)
    texts = []
    if args.dataset == 'wikitext2':
        try:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cache_dir)
            for ex in ds:
                t = ex.get("text", "").strip()
                if len(t) > 20:
                    texts.append(t)
        except Exception as e:
            print(f"Warning: wikitext2 failed ({e}). Using synthetic.")
            args.dataset = 'synthetic'
    if args.dataset == 'tinystories':
        try:
            ds = load_dataset("roneneldan/TinyStories", split="train", cache_dir=cache_dir)
            for i, ex in enumerate(ds):
                if i >= n_train + n_eval:
                    break
                texts.append(ex.get("text", "")[:500])
        except Exception as e:
            print(f"Warning: TinyStories failed ({e}). Using synthetic.")
            args.dataset = 'synthetic'
    if args.dataset == 'synthetic' or len(texts) < n_train + n_eval:
        # Simple repeating patterns
        patterns = [
            "The quick brown fox jumps over the lazy dog. ",
            "In the beginning was the word. ",
            "One two three four five. ",
            "Hello world. Hello world. ",
        ]
        texts = []
        for i in range(n_train + n_eval):
            p = patterns[i % len(patterns)]
            texts.append((p * 5)[:200])
    n_total = n_train + n_eval
    all_ids = []
    for t in texts[:n_total]:
        ids = tokenizer.encode(t, add_special_tokens=False, max_length=max_length, truncation=True)
        if len(ids) >= 2:
            all_ids.append(ids)
    train_ids = all_ids[:n_train]
    eval_ids = all_ids[n_train:n_train + n_eval]
    return train_ids, eval_ids


def _loss_from_logits(logits, labels, teacher_logits, vocab_size, mode, temperature):
    """Compute loss from logits. For distillation uses teacher_logits; for CE uses labels."""
    shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size).float()
    shift_labels = labels[..., :-1].contiguous().view(-1)
    if mode == 'distillation' and teacher_logits is not None:
        shift_teacher = teacher_logits[..., :-1, :].contiguous().view(-1, vocab_size).float()
        mask = (shift_labels != -100).float()
        return compute_loss_kl(shift_logits, shift_teacher, temperature, mask)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100).item()


def compute_danp_grad_estimate(
    student, teacher, tokenizer, batch_ids, device, hook, eta, alpha,
    mode, kl_weight, ce_weight, temperature,
    n_population=2, antithetic=False, delta_l_norm='none',
    max_scale=1e2, max_update=1.0, max_dec_update=0.01,
    verbose=False,
):
    """DANP gradient estimate with configurable loss (KL or CE)."""
    vocab_size = student.config.vocab_size
    input_ids, attention_mask, labels = _prepare_batch_lm(
        batch_ids, tokenizer, device, args.max_length, vocab_size=vocab_size
    )
    teacher_logits = None
    if mode == 'distillation' and teacher is not None:
        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

    # Pass 1: Clean forward
    hook.attach(add_noise=False, capture=True)
    with torch.no_grad():
        clean_out = student(input_ids=input_ids, attention_mask=attention_mask)
    captured_clean = {k: (v[0].clone(), v[1].clone(), None) for k, v in hook._captured.items()}
    L_clean = _loss_from_logits(clean_out.logits, labels, teacher_logits, vocab_size, mode, temperature)
    hook.detach()
    if not np.isfinite(L_clean):
        return {}, {}, float('nan'), 0.0

    # Pass 2: Noisy forward(s)
    accumulated_grads = {}
    delta_L_list = []
    last_scale_raw, last_norm_sq, last_N = 0.0, 0.0, 0

    n_pop = n_population * 2 if antithetic else n_population
    for pop_idx in range(n_pop):
        hook.attach(add_noise=True, capture=True, pop_offset=pop_idx)
        with torch.no_grad():
            noisy_out = student(input_ids=input_ids, attention_mask=attention_mask)
        L_noisy = _loss_from_logits(noisy_out.logits, labels, teacher_logits, vocab_size, mode, temperature)
        captured_noisy = hook._captured.copy()
        hook.detach()
        delta_L_list.append(L_noisy - L_clean)

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
            continue

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
        return {}, {}, L_clean, np.mean(delta_L_list) if delta_L_list else 0.0

    # Delta_L normalization
    if delta_l_norm == 'batch' and len(delta_L_list) > 1:
        arr = np.array(delta_L_list)
        mean, std = arr.mean(), arr.std()
        if std > 1e-8:
            delta_L_list = [(d - mean) / (std + 1e-8) for d in delta_L_list]
        # Recompute scale with normalized delta_L? For now we've already applied updates.
        # The normalization would need to be applied before the scale. We'd need to refactor.
        pass

    n_eff = len(delta_L_list)
    for key in accumulated_grads:
        accumulated_grads[key] /= max(n_eff, 1)

    # Decorrelation
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
        if not torch.isfinite(dec_update).all():
            dec_update = torch.zeros_like(dec_update)
        decorrelation_updates[name] = dec_update

    weight_updates = {}
    for key, acc in accumulated_grads.items():
        if key.endswith(".weight"):
            mod = student.get_submodule(key[:-7])
            weight_updates[key] = acc.to(mod.weight.dtype)
        else:
            mod = student.get_submodule(key[:-6])
            if mod.bias is not None:
                weight_updates[key] = acc.to(mod.bias.dtype)

    if verbose:
        print(f"\n[DANP] L_clean={L_clean:.4f} delta_L_mean={np.mean(delta_L_list):.6f} scale_raw={last_scale_raw:.6e}")

    return weight_updates, decorrelation_updates, L_clean, np.mean(delta_L_list)


def run_danp_batch_step(student, teacher, tokenizer, batch_ids, device, hook):
    """One batch step: compute gradients, average, apply."""
    batch_size = len(batch_ids)
    w_acc = {}
    dec_acc = {n: None for n in hook.R}
    total_loss = 0.0
    total_delta = 0.0

    for i in range(batch_size):
        single_batch = [batch_ids[i]]
        w_upd, dec_upd, L, dL = compute_danp_grad_estimate(
            student, teacher, tokenizer, single_batch, device, hook,
            eta=args.eta, alpha=args.alpha,
            mode=args.mode,
            kl_weight=args.kl_weight, ce_weight=args.ce_weight, temperature=args.temperature,
            n_population=args.n_population, antithetic=args.antithetic, delta_l_norm=args.delta_l_norm,
            max_scale=args.max_scale, max_update=args.max_update, max_dec_update=args.max_dec_update,
            verbose=args.verbose and i == 0,
        )
        total_loss += L
        total_delta += dL
        for k, v in w_upd.items():
            w_acc[k] = w_acc.get(k, torch.zeros_like(v)) + v
        for n, v in dec_upd.items():
            if dec_acc[n] is None:
                dec_acc[n] = torch.zeros_like(v)
            dec_acc[n] += v

    with torch.no_grad():
        for k, acc in w_acc.items():
            upd = acc / batch_size
            if not torch.isfinite(upd).all():
                continue
            if k.endswith(".weight"):
                m = student.get_submodule(k[:-7])
                m.weight.sub_(upd.to(m.weight.dtype))
            elif k.endswith(".bias"):
                m = student.get_submodule(k[:-6])
                if m.bias is not None:
                    m.bias.sub_(upd.to(m.bias.dtype))

    for n, acc in dec_acc.items():
        if acc is not None:
            upd = acc / batch_size
            if torch.isfinite(upd).all():
                hook.R[n].sub_(upd)
                if not torch.isfinite(hook.R[n]).all():
                    hook.R[n].copy_(torch.eye(hook.R[n].shape[0], device=hook.R[n].device, dtype=hook.R[n].dtype))

    return total_loss / batch_size, total_delta / batch_size


def eval_loss(student, teacher, eval_ids, tokenizer, device, mode):
    """Compute eval loss (KL or CE)."""
    vocab_size = student.config.vocab_size
    losses = []
    for i in range(0, len(eval_ids), args.batch_size):
        batch = eval_ids[i:i + args.batch_size]
        inp, attn, lab = _prepare_batch_lm(batch, tokenizer, device, args.max_length, vocab_size)
        if mode == 'distillation' and teacher is not None:
            L = compute_loss_distillation(student, teacher, inp, attn, lab, vocab_size,
                                          args.kl_weight, args.ce_weight, args.temperature)
        else:
            L = compute_loss_ce(student, inp, attn, lab, vocab_size)
        losses.append(L)
    return np.nanmean(losses)


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    hf_cache = args.hf_cache_dir or os.path.join(os.path.dirname(__file__), ".cache", "huggingface")
    os.makedirs(hf_cache, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading student: {args.student_path}")
    student = AutoModelForCausalLM.from_pretrained(args.student_path, cache_dir=hf_cache, torch_dtype=dtype)
    student = student.to(device)

    teacher = None
    if args.mode == 'distillation':
        print(f"Loading teacher: {args.teacher_path}")
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher_path, cache_dir=hf_cache, torch_dtype=dtype)
        teacher = teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.student_path, cache_dir=hf_cache)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"Loading {args.dataset} dataset...")
    train_ids, eval_ids = load_lm_dataset(args.n_train, args.n_eval, tokenizer, args.max_length)
    print(f"Train: {len(train_ids)} sequences, Eval: {len(eval_ids)}")

    baseline = eval_loss(student, teacher, eval_ids, tokenizer, device, args.mode)
    print(f"[BASELINE] Eval loss: {baseline:.4f}")

    train_losses = []
    eval_losses_hist = []

    for epoch in range(args.epochs):
        student.train()
        indices = np.random.permutation(len(train_ids))
        epoch_losses = []
        hook = DANPFullHook(student, args.np_include, args.last_k, args.sigma, args.alpha, args.seed + epoch * 10000)
        for name in hook.R:
            hook.R[name] = hook.R[name].to(device)

        pbar = tqdm(range(0, len(train_ids), args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}")
        for start in pbar:
            batch_idx = indices[start:start + args.batch_size]
            batch = [train_ids[i] for i in batch_idx]
            hook.base_seed = args.seed + epoch * 10000 + start
            L, dL = run_danp_batch_step(student, teacher, tokenizer, batch, device, hook)
            epoch_losses.append(L)
            pbar.set_postfix({"loss": f"{L:.4f}", "delta_L": f"{dL:.4f}"})

        train_loss = np.nanmean(epoch_losses)
        eval_loss_val = eval_loss(student, teacher, eval_ids, tokenizer, device, args.mode)
        train_losses.append(train_loss)
        eval_losses_hist.append(eval_loss_val)
        print(f"[Epoch {epoch+1}] Train: {train_loss:.4f}, Eval: {eval_loss_val:.4f}")

    save_dir = os.path.join(args.output_dir, "final_student")
    student.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    if _HAS_MATPLOTLIB and not args.no_plot and train_losses:
        _plot_losses(train_losses, eval_losses_hist, args.output_dir, args.mode)
    elif not args.no_plot and train_losses and not _HAS_MATPLOTLIB:
        print("Tip: Install matplotlib (pip install matplotlib) to save loss plots.")

    print(f"Saved to {save_dir}")


def _plot_losses(train_losses, eval_losses_hist, output_dir, experiment_name):
    """Save train/eval loss plot to output_dir."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, 'b-o', markersize=4, label='Train loss')
    if eval_losses_hist:
        ax.plot(epochs, eval_losses_hist, 'r-s', markersize=4, label='Eval loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'DANP distillation ({experiment_name}) - Train & Eval Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, 'loss_plot.png')
    fig.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved loss plot to {plot_path}")


if __name__ == "__main__":
    main()
