import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import copy
import os
import argparse
import re
import hashlib
from accelerate import Accelerator
import time
import torch.multiprocessing as mp
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import math
import gc

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='huggingface_cache')
parser.add_argument('--precision', type=str, default='bf16')
parser.add_argument('--gpu_threads', type=int, default=4, help='Number of parallel threads per GPU')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
parser.add_argument('--aggressive_gc', action='store_true', help='Perform aggressive garbage collection')
parser.add_argument('--eval_interval', type=int, default=20, help='Interval for evaluating best/worst models')
parser.add_argument('--visualization_dir', type=str, default='./visualizations', help='Directory for saving visualizations')
parser.add_argument('--weight_sample_interval', type=int, default=10, help='Sample interval for weight tracking')
parser.add_argument('--algorithm', type=str, default='danp', choices=['wp', 'danp'])
parser.add_argument('--eta', type=float, default=1e-5, help='DANP learning rate')
parser.add_argument('--sigma', type=float, default=1e-2, help='DANP activity noise std')
parser.add_argument('--alpha', type=float, default=0.0, help='DANP decorrelation strength')
parser.add_argument('--n_population', type=int, default=8, help='DANP population size per sample')
parser.add_argument('--np_include', type=str, default='mlp', choices=['head', 'attn', 'mlp', 'all'])
parser.add_argument('--max_length', type=int, default=256)
args = parser.parse_args()


NUM_ITERATIONS = 1000              # Number of ES iterations (generations)
POPULATION_SIZE = 30              # Population size (WP only)
SIGMA = 0.001                     # Standard deviation for WP weight perturbations
ALPHA = 0.0005                    # Learning rate (WP only)
max_new_tokens = 100
do_sample = False
initial_seed = 33


# --- Dataset ---
dataset = [
    ("Solve: 3 + 5 =", "8"),
    ("If all birds can fly and penguins are birds, can penguins fly?", "No"),
]


# --- DANP infrastructure ---
_ARCH_PATTERNS = [(r'\.layers\.(\d+)\.', 'layers'), (r'\.h\.(\d+)\.', 'h')]

def _detect_layer_pattern(model):
    for pattern, _ in _ARCH_PATTERNS:
        names = [n for n, _ in model.named_modules() if re.search(pattern, n)]
        if names:
            idxs = [int(re.search(pattern, n).group(1)) for n in names]
            return pattern, max(idxs) + 1
    return None, 0

def _stable_uid(s):
    h = hashlib.blake2s(s.encode(), digest_size=8).digest()
    return int.from_bytes(h[:4], 'little') & 0x7FFFFFFF

def _infer_total_blocks(model):
    return _detect_layer_pattern(model)[1]

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
    if include == 'mlp':
        return any(k in lname for k in ('mlp', 'ffn', 'gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2', 'c_fc', 'c_proj'))
    if include == 'attn':
        return any(k in lname for k in ('attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj'))
    return include == 'all'

def _get_target_linears_ordered(model, include, last_k=0):
    total = _infer_total_blocks(model)
    return [(n, m) for n, m in model.named_modules()
            if _is_target_linear(include, n, m, last_k, total)]

def _get_R_in_producer(name, in_features, targets_by_name):
    if "down_proj" in name:
        producer = name.replace("down_proj", "gate_proj")
    elif "gate_proj" in name or "up_proj" in name:
        producer = name.replace("mlp.gate_proj", "self_attn.o_proj").replace("mlp.up_proj", "self_attn.o_proj")
    elif "c_proj" in name:
        producer = name.replace("c_proj", "c_fc")
    else:
        return None
    if producer in targets_by_name and targets_by_name[producer].out_features == in_features:
        return producer
    return None

class DANPFullHook:
    def __init__(self, model, include, last_k, sigma, alpha, base_seed):
        self.model = model
        self.include, self.last_k = include, last_k
        self.sigma, self.alpha = float(sigma), float(alpha)
        self.base_seed = int(base_seed)
        self._orig, self._targets = {}, _get_target_linears_ordered(model, include, last_k)
        self._layer_uids = {n: _stable_uid(n) for n, _ in self._targets}
        self._row_counters = {n: 0 for n, _ in self._targets}
        self.R, self._R_in = {}, {}
        targets_by_name = dict(self._targets)
        # Keep R on CPU to avoid OOM on large models; move to device lazily in forward
        for name, mod in self._targets:
            R_out = torch.eye(mod.out_features, dtype=torch.float32, device=torch.device('cpu'))
            self.R[name] = R_out
            producer = _get_R_in_producer(name, mod.in_features, targets_by_name)
            self._R_in[name] = self.R[producer] if producer else None
        self._add_noise, self._capture, self._captured, self._pop_offset = True, False, {}, 0

    def _seed_for_call(self, layer_uid, row_start):
        return ((self.base_seed ^ layer_uid) + 0x9e3779b9 + (row_start * 2654435761 & 0x7FFFFFFF)
                + (self._pop_offset * 1000000)) & 0x7FFFFFFF

    def attach(self, add_noise=True, capture=False, pop_offset=0):
        self._add_noise, self._capture, self._captured, self._pop_offset = add_noise, capture, {}, pop_offset
        for n in self._row_counters:
            self._row_counters[n] = 0
        for name, mod in self._targets:
            self._wrap(name, mod)

    def detach(self):
        for name, mod in self._targets:
            if name in self._orig:
                mod.forward = self._orig[name]
        self._orig.clear()

    def _wrap(self, name, mod):
        orig = mod.forward
        R_in, R_out = self._R_in[name], self.R[name]
        layer_uid = self._layer_uids[name]

        def fwd(x):
            dev = x.device
            x2 = x.float().reshape(-1, x.shape[-1]) if x.dim() > 2 else x.float()
            R_in_dev = self._R_in[name].to(dev).to(x2.dtype) if self._R_in[name] is not None else None
            x_star = x2 @ R_in_dev.T if R_in_dev is not None else x2
            with torch.no_grad():
                y = orig(x_star.to(mod.weight.dtype)).float()
                g = torch.Generator(device=y.device)
                g.manual_seed(self._seed_for_call(layer_uid, self._row_counters[name]))
                xi = torch.randn_like(y, generator=g) * self.sigma
                y_noisy = y + xi if self._add_noise else y
                if self._capture:
                    self._captured[name] = (x_star.detach().clone(), y.detach().clone(),
                                          y_noisy.detach().clone() if self._add_noise else None)
                self._row_counters[name] += x2.shape[0]
                R_out_dev = self.R[name].to(dev).to(y_noisy.dtype)
                out = (y_noisy @ R_out_dev.T).to(mod.weight.dtype)
            if x.dim() > 2:
                out = out.reshape(*x.shape[:-1], out.shape[-1])
            return out
        mod.forward = fwd
        self._orig[name] = orig

def _prepare_batch(batch, tokenizer, device, max_length, vocab_size=None, label_ignore_id=-100):
    input_ids_list, labels_list = [], []
    for p, t in batch:
        prompt_ids = tokenizer.encode(p, add_special_tokens=True)
        target_ids = tokenizer.encode(t, add_special_tokens=False)
        full = (prompt_ids + target_ids)[:max_length]
        lab = [label_ignore_id] * len(prompt_ids) + target_ids[:max_length]
        lab = lab[:max_length]
        input_ids_list.append(full)
        labels_list.append(lab)
    pad_id = tokenizer.pad_token_id or 0
    max_len = max(len(x) for x in input_ids_list)
    inp = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
    lab = torch.full((len(batch), max_len), label_ignore_id, dtype=torch.long, device=device)
    for i in range(len(batch)):
        L = len(input_ids_list[i])
        ids = torch.tensor(input_ids_list[i], device=device)
        lbs = torch.tensor(labels_list[i], device=device)
        if vocab_size:
            ids = ids.clamp(0, vocab_size - 1)
            lbs = torch.where(lbs == label_ignore_id, lbs, lbs.clamp(0, vocab_size - 1))
        inp[i, :L], attn[i, :L], lab[i, :L] = ids, 1, lbs
    return inp, attn, lab

def compute_loss_danp(model, tokenizer, batch, device, max_length, vocab_size):
    inp, attn, lab = _prepare_batch(batch, tokenizer, device, max_length, vocab_size)
    with torch.no_grad():
        logits = model(input_ids=inp, attention_mask=attn).logits.float().view(-1, vocab_size).clamp(-50, 50)
        return F.cross_entropy(logits, lab.view(-1), ignore_index=-100)

def compute_danp_grad_per_sample(model, tokenizer, example, device, hook, eta, alpha, n_population,
                                  max_length, vocab_size, max_scale=1e2, max_update=1.0, max_dec_update=0.01):
    batch = [example]
    inp, attn, lab = _prepare_batch(batch, tokenizer, device, max_length, vocab_size)
    hook.attach(add_noise=False, capture=True)
    with torch.no_grad():
        logits = model(input_ids=inp, attention_mask=attn).logits.float().view(-1, vocab_size).clamp(-50, 50)
        L_clean = F.cross_entropy(logits, lab.view(-1), ignore_index=-100).item()
    if not np.isfinite(L_clean):
        hook.detach()
        return {}, {}, L_clean, 0.0
    captured_clean = {k: (v[0].clone(), v[1].clone(), None) for k, v in hook._captured.items()}
    hook.detach()

    acc_grads, delta_L_sum = {}, 0.0
    for pop_idx in range(n_population):
        hook.attach(add_noise=True, capture=True, pop_offset=pop_idx)
        with torch.no_grad():
            logits = model(input_ids=inp, attention_mask=attn).logits.float().view(-1, vocab_size).clamp(-50, 50)
            L_noisy = F.cross_entropy(logits, lab.view(-1), ignore_index=-100).item()
        captured_noisy = hook._captured.copy()
        hook.detach()
        delta_L_sum += L_noisy - L_clean

        delta_a_all = []
        for name, _ in hook._targets:
            if name in captured_clean and name in captured_noisy and captured_noisy[name][2] is not None:
                delta_a_all.append((captured_noisy[name][2] - captured_clean[name][1]).flatten())
        if not delta_a_all:
            continue
        delta_a_cat = torch.cat(delta_a_all)
        norm_sq = (delta_a_cat ** 2).sum().item() + 1e-8
        N = delta_a_cat.numel()
        delta_L = np.clip(L_noisy - L_clean, -1e4, 1e4)
        if not np.isfinite(delta_L):
            delta_L = 0.0
        scale = np.clip(eta * N * delta_L / norm_sq, -max_scale, max_scale)

        for name, mod in hook._targets:
            if name not in captured_clean or name not in captured_noisy or captured_noisy[name][2] is None:
                continue
            x_star, y_clean, _ = captured_clean[name]
            _, _, y_noisy = captured_noisy[name]
            delta_a = y_noisy - y_clean
            x_star = x_star.reshape(-1, x_star.shape[-1]) if x_star.dim() > 2 else x_star
            delta_a = delta_a.reshape(-1, delta_a.shape[-1]) if delta_a.dim() > 2 else delta_a
            grad = delta_a.T @ x_star
            if not torch.isfinite(grad).all():
                continue
            upd = torch.clamp(scale * grad, -max_update, max_update)
            key = name + ".weight"
            if key not in acc_grads:
                acc_grads[key] = torch.zeros_like(grad, device=grad.device)
            acc_grads[key].add_(upd)

    if not acc_grads:
        return {}, {}, L_clean, delta_L_sum / max(1, n_population)

    weight_updates = {k: v / n_population for k, v in acc_grads.items()}
    decorrelation_updates = {}
    for name, _ in hook._targets:
        if name not in captured_clean:
            continue
        _, y_clean, _ = captured_clean[name]
        R_out = hook.R[name].to(y_clean.device)
        x_star_out = (y_clean @ R_out.T).reshape(-1, R_out.shape[0])
        cov = x_star_out.T @ x_star_out / max(x_star_out.shape[0], 1)
        diag = torch.diag((x_star_out ** 2).mean(dim=0))
        dec_upd = torch.clamp(alpha * (cov - diag) @ R_out, -max_dec_update, max_dec_update)
        decorrelation_updates[name] = dec_upd if torch.isfinite(dec_upd).all() else torch.zeros_like(R_out)

    return weight_updates, decorrelation_updates, L_clean, delta_L_sum / n_population

def run_danp_batch_step(model, tokenizer, batch, device, hook, eta, alpha, n_population, max_length, vocab_size, verbose=False):
    acc_weight, acc_decorr = {}, {n: None for n in hook.R}
    total_loss, total_delta = 0.0, 0.0
    for example in batch:
        wu, du, Lc, dL = compute_danp_grad_per_sample(
            model, tokenizer, example, device, hook, eta, alpha, n_population,
            max_length, vocab_size,
        )
        total_loss += Lc
        total_delta += dL
        for k, v in wu.items():
            acc_weight[k] = acc_weight.get(k, torch.zeros_like(v)) + v
        for n, v in du.items():
            if acc_decorr[n] is None:
                acc_decorr[n] = torch.zeros_like(v)
            acc_decorr[n] += v

    batch_size = len(batch)
    with torch.no_grad():
        for k, acc in acc_weight.items():
            upd = acc / batch_size
            if not torch.isfinite(upd).all():
                continue
            mod = model.get_submodule(k[:-7])
            mod.weight.sub_(upd.to(mod.weight.dtype))
        for n, acc in acc_decorr.items():
            if acc is not None:
                upd = (acc / batch_size).cpu()
                if torch.isfinite(upd).all():
                    hook.R[n].sub_(upd)
                    if not torch.isfinite(hook.R[n]).all():
                        hook.R[n].copy_(torch.eye(hook.R[n].shape[0], device=torch.device('cpu'), dtype=hook.R[n].dtype))
    return total_loss / batch_size, total_delta / batch_size


def compute_reward(generated_text, target_text):
    return -abs(len(generated_text) - len(target_text))

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def evaluate_model(model, tokenizer, input_text, target_text, accelerator, seed_idx=None, thread_id=None, verbose=False, return_text=False):
    """
    Generate a response from the model given an input (single or batch) and compute rewards.
    """
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} evaluating seed {seed_idx}")

    # Handle both single input and batch input
    is_batch = isinstance(input_text, list)
    input_texts = input_text if is_batch else [input_text]
    target_texts = target_text if is_batch else [target_text]

    # Batch tokenization
    tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = tokenized_inputs["input_ids"].to(accelerator.device)
    attention_mask = tokenized_inputs["attention_mask"].to(accelerator.device)
    with torch.inference_mode():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=do_sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

    # Decode batch outputs
    generated_texts = []
    for i in range(len(input_texts)):
        try:
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(outputs[i], skip_special_tokens=True)
            filtered = [t for t in tokens if t is not None]
            generated_text = tokenizer.convert_tokens_to_string(filtered)
        generated_texts.append(generated_text)

    del input_ids, outputs
    torch.cuda.empty_cache()

    # Compute rewards for batch texts
    rewards = [compute_reward(gen_text, tgt_text) for gen_text, tgt_text in zip(generated_texts, target_texts)]


    if return_text:
        return rewards, generated_texts
    else:
        return rewards

def process_seed(seed_args):
    """Function to process a single seed, used for thread pool"""
    seed_idx, seed, model, tokenizer, accelerator, thread_id, verbose = seed_args

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed})")

    # Put load-evaluate-restore in the same lock block for thread safety
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)

        gen.manual_seed(int(seed))

        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(SIGMA * noise)

    # Ensure weights are fully loaded before evaluation
    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    # Evaluate all prompts with perturbed weights in batch
    input_texts = [input_text for input_text, _ in dataset]
    target_texts = [target_text for _, target_text in dataset]
    rewards = evaluate_model(model, tokenizer, input_texts, target_texts, accelerator,
                           seed_idx=seed_idx, thread_id=thread_id, verbose=verbose, return_text=False)
    total_reward = sum(rewards)

    # Restore original weights (direct inplace modification)
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)

        gen.manual_seed(int(seed))

        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(-SIGMA * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    average_reward = total_reward / len(dataset)


    force_memory_cleanup()

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} completed seed {seed_idx} with reward {average_reward:.4f}")

    return seed_idx, average_reward


# --- Main Evolution Strategies Loop ---
def main():
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print(f"Algorithm: {args.algorithm}, Total processes: {accelerator.num_processes}, GPU threads: {args.gpu_threads}")
        print(f"Iterations: {NUM_ITERATIONS}")
        if args.algorithm == 'danp':
            print(f"eta: {args.eta}, sigma: {args.sigma}, alpha: {args.alpha}, n_pop: {args.n_population}")
        else:
            print(f"Sigma: {SIGMA}, Alpha: {ALPHA}, Population: {POPULATION_SIZE}")
        print(f"Visualization directory: {args.visualization_dir}")

    # Load model
    model_name = args.model_name
    hf_cache_dir = args.hf_cache_dir

    if accelerator.is_main_process:
        print(f"Loading model {model_name}...")


    # Load model on main process first then sync
    model_list = []
    for model_index in range(args.gpu_threads):
        model_list.append(AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            device_map={"": accelerator.process_index},  # Assign devices explicitly
            torch_dtype=torch.float16 if args.precision == 'fp16' else (torch.bfloat16 if args.precision == 'bf16' else torch.float32),
        ))
        # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)

    if accelerator.is_main_process:
        print("Model loaded successfully")

    # Prepare model with accelerator
    for model in model_list:
        model.eval()  # Turn off dropout, etc.

    force_memory_cleanup()

    # Record total training start time
    training_start_time = time.time()

    np.random.seed(initial_seed)

    # DANP: set tokenizer for batch preparation
    if args.algorithm == 'danp':
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
        vocab_size = model_list[0].config.vocab_size

    for iteration in range(NUM_ITERATIONS):
        iter_start_time = time.time()
        force_memory_cleanup()

        if args.algorithm == 'danp':
            # --- DANP iteration: loss-based, activity perturbation ---
            if args.verbose:
                print(f"Process {accelerator.process_index} DANP iteration {iteration + 1}/{NUM_ITERATIONS}")
            np.random.seed(initial_seed + iteration)
            torch.manual_seed(initial_seed + iteration)
            model = model_list[0]
            model.train()
            hook = DANPFullHook(model, args.np_include, 0, args.sigma, args.alpha, initial_seed + iteration)
            # R matrices stay on CPU; moved to device lazily in forward to avoid OOM

            batch = list(dataset)
            L_clean, delta_L = run_danp_batch_step(
                model, tokenizer, batch, accelerator.device, hook,
                eta=args.eta, alpha=args.alpha, n_population=args.n_population,
                max_length=args.max_length, vocab_size=vocab_size, verbose=args.verbose,
            )
            for m in model_list[1:]:
                for n, p in m.named_parameters():
                    p.data.copy_(model.get_parameter(n).data.clone())

        else:
            # --- WP iteration: reward-based, weight perturbation ---
            if args.verbose:
                print(f"Process {accelerator.process_index} starting iteration {iteration + 1}/{NUM_ITERATIONS}")
            if accelerator.is_main_process:
                seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()
                seeds_tensor = torch.tensor(seeds, device=accelerator.device)
            else:
                seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=accelerator.device)
            torch.distributed.broadcast(seeds_tensor, src=0)
            seeds = seeds_tensor.cpu().tolist()

            local_seeds = [(i, s) for i, s in enumerate(seeds) if i % accelerator.num_processes == accelerator.process_index]
            local_rewards = []
            batch_size = max(1, min(args.gpu_threads, len(local_seeds)))
            for batch_start in range(0, len(local_seeds), batch_size):
                batch_seeds = local_seeds[batch_start:batch_start + batch_size]
                with ThreadPoolExecutor(max_workers=len(batch_seeds)) as executor:
                    thread_args = [(idx, seed, model_list[t], tokenizer, accelerator, t, args.verbose)
                                  for t, (idx, seed) in enumerate(batch_seeds)]
                    local_rewards.extend(list(executor.map(process_seed, thread_args)))
                force_memory_cleanup()

            all_rewards = torch.zeros(POPULATION_SIZE, device=accelerator.device)
            for seed_idx, reward in local_rewards:
                all_rewards[seed_idx] = reward
            torch.distributed.all_reduce(all_rewards, op=torch.distributed.ReduceOp.SUM)
            rewards = all_rewards.cpu().tolist()
            del all_rewards
            force_memory_cleanup()

            rewards_tensor = np.array(rewards, dtype=np.float32)
            rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

            original_model = model_list[0]
            for name, param in original_model.named_parameters():
                gen = torch.Generator(device=param.device)
                update = torch.zeros_like(param)
                for seed_idx in range(POPULATION_SIZE):
                    gen.manual_seed(int(seeds[seed_idx]))
                    noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
                    update.add_(noise.mul_(float(rewards_normalized[seed_idx])))
                param.data.add_(ALPHA * update.div_(POPULATION_SIZE))
                torch.cuda.empty_cache()
            for m in model_list[1:]:
                for n, p in m.named_parameters():
                    p.data.copy_(original_model.get_parameter(n).data.clone())

        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)
        force_memory_cleanup()

        iter_time = time.time() - iter_start_time
        if accelerator.is_main_process:
            if args.algorithm == 'danp':
                print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, Loss: {L_clean:.4f}, delta_L: {delta_L:.4f}")
            else:
                mr, mn, mx = rewards_tensor.mean(), rewards_tensor.min(), rewards_tensor.max()
                print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, Mean: {mr:.2f}, Min: {mn:.2f}, Max: {mx:.2f}")
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated")

    total_time = time.time() - training_start_time


    # Save the fine-tuned model weights.
    if accelerator.is_main_process:
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        question_num = len(dataset)
        if args.algorithm == 'danp':
            save_dir = f"{model_name}_danp_seed{initial_seed}_iter{NUM_ITERATIONS}_eta{args.eta}_sig{args.sigma}_alpha{args.alpha}_npop{args.n_population}_{args.precision}_n{question_num}"
        else:
            save_dir = f"{model_name}_es_random_seed{initial_seed}_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_threads{args.gpu_threads}_question_num{question_num}_correct"
        print(f"Saving model to {save_dir}...")
        original_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model saved successfully.")
        print(f"Visualizations saved to {args.visualization_dir}")

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()