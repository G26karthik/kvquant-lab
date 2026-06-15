"""
wht_quantizer.py
================
Improved TurboQuant using Walsh-Hadamard Transform (WHT) and Asymmetric QJL.

Features:
  1. O(d log d) recursive butterfly WHT instead of O(d^2) QR.
  2. Asymmetric QJL: 2-stage (MSE + QJL) on Key, MSE-only on Value.
  3. Saves parameter storage and norm memory, speeding up quantization.
"""

import os
import gc
import json
import math
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from math import lgamma
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Config
SEED = 42
MODEL_NAME = "gpt2-medium"
MAX_NEW_TOKENS = 50
QUANT_BITS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ----------------------------------------------------------------------
# WHT Butterfly Transform
# ----------------------------------------------------------------------
def fwht_pytorch(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    d = shape[-1]
    x = x.clone().reshape(-1, d)
    h = 1
    while h < d:
        x = x.reshape(-1, d // (2 * h), 2 * h)
        x_left = x[..., :h]
        x_right = x[..., h:]
        x = torch.cat([x_left + x_right, x_left - x_right], dim=-1)
        h *= 2
    return x.reshape(shape) / math.sqrt(d)


# ----------------------------------------------------------------------
# Lloyd-Max Codebook Computation
# ----------------------------------------------------------------------
def compute_lloyd_max_codebook(dim: int, n_levels: int,
                                n_iter: int = 300,
                                n_grid: int = 80000) -> np.ndarray:
    x = np.linspace(-1 + 1e-10, 1 - 1e-10, n_grid)
    dx = x[1] - x[0]

    log_const = lgamma(dim / 2) - 0.5 * np.log(np.pi) - lgamma((dim - 1) / 2)
    log_pdf = log_const + ((dim - 3) / 2) * np.log(np.maximum(1 - x ** 2, 1e-30))
    pdf = np.exp(log_pdf)
    pdf = pdf / (np.sum(pdf) * dx)

    cdf = np.cumsum(pdf) * dx
    cdf = cdf / cdf[-1]

    centroids = np.zeros(n_levels)
    for i in range(n_levels):
        target = (i + 0.5) / n_levels
        idx = np.searchsorted(cdf, target)
        centroids[i] = x[min(idx, n_grid - 1)]

    for _ in range(n_iter):
        bounds = np.concatenate([[x[0]], (centroids[:-1] + centroids[1:]) / 2, [x[-1]]])
        new_centroids = np.zeros(n_levels)
        for i in range(n_levels):
            lo, hi = bounds[i], bounds[i + 1]
            mask = (x >= lo) & (x <= hi)
            w = pdf[mask]
            if w.sum() > 1e-30:
                new_centroids[i] = np.average(x[mask], weights=w)
            else:
                new_centroids[i] = centroids[i]
        if np.allclose(centroids, new_centroids, atol=1e-12):
            break
        centroids = new_centroids
    return np.sort(centroids)


# ----------------------------------------------------------------------
# WHT-based TurboQuantMSE
# ----------------------------------------------------------------------
class TurboQuantMSE:
    def __init__(self, dim: int, bits: int, device: str, seed: int = 0):
        self.dim = dim
        self.bits = bits
        self.n_levels = 2 ** bits
        self.device = device

        # Preconditioning random ±1 signs
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        self.signs = (torch.randint(0, 2, (dim,), generator=rng).float() * 2 - 1).to(device).to(DTYPE)

        # Precomputed Lloyd-Max codebook
        codebook_np = compute_lloyd_max_codebook(dim, self.n_levels)
        self.codebook = torch.tensor(codebook_np, dtype=torch.float32, device=device)
        self.boundaries = (self.codebook[:-1] + self.codebook[1:]) / 2

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_f32 = x.float()
        norms = torch.norm(x_f32, dim=-1, keepdim=True)
        x_hat = x_f32 / (norms + 1e-8)

        # WHT + sign flip rotation
        y = fwht_pytorch(x_hat.to(DTYPE) * self.signs).float()
        indices = torch.bucketize(y, self.boundaries).to(torch.int8)
        return indices, norms

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        y_hat = self.codebook[indices.long()]
        # Derotate: self-inverse WHT + sign flip
        x_hat = fwht_pytorch(y_hat.to(DTYPE)) * self.signs
        x_hat = x_hat * norms.to(DTYPE)
        return x_hat.half()


# ----------------------------------------------------------------------
# WHT-based TurboQuantProd
# ----------------------------------------------------------------------
class TurboQuantProd:
    def __init__(self, dim: int, bits: int, device: str, seed: int = 0):
        self.dim = dim
        self.bits = bits
        self.device = device

        # Stage 1: MSE quantizer with (b-1) bits
        self.mse_quantizer = TurboQuantMSE(dim, bits - 1, device, seed=seed)

        # Stage 2: QJL random projection matrix S
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed + 10000)
        self.S = torch.randn(dim, dim, generator=rng, dtype=torch.float32).to(device)
        self.qjl_scale = math.sqrt(math.pi / 2) / dim

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices, norms = self.mse_quantizer.quantize(x)
        x_mse = self.mse_quantizer.dequantize(indices, norms).float()

        r = x.float() - x_mse
        gamma = torch.norm(r, dim=-1, keepdim=True)
        r_hat = r / (gamma + 1e-8)

        # QJL projection
        projection = r_hat @ self.S.T
        qjl_signs = torch.sign(projection)
        qjl_signs[qjl_signs == 0] = 1
        qjl_signs = qjl_signs.to(torch.int8)

        return indices, norms, qjl_signs, gamma

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor,
                    qjl_signs: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        x_mse = self.mse_quantizer.dequantize(indices, norms).float()
        x_qjl = gamma * self.qjl_scale * (qjl_signs.float() @ self.S)
        return (x_mse + x_qjl).half()


# ----------------------------------------------------------------------
# Adaptive QJL (Section 4.2 dimension-aware application)
# ----------------------------------------------------------------------
class AdaptiveTurboQuantProd:
    def __init__(self, dim: int, bits: int, device: str, seed: int = 0):
        self.dim = dim
        self.bits = bits
        self.device = device
        
        if dim >= 128:
            self.prod_quantizer = TurboQuantProd(dim, bits, device, seed=seed)
            self.use_qjl = True
        else:
            self.mse_quantizer = TurboQuantMSE(dim, bits, device, seed=seed)
            self.use_qjl = False
            print("QJL disabled at d=64 (effective threshold: d>=128)")

    def quantize(self, x: torch.Tensor):
        if self.use_qjl:
            return self.prod_quantizer.quantize(x)
        else:
            indices, norms = self.mse_quantizer.quantize(x)
            return (indices, norms, None, None)

    def dequantize(self, *args) -> torch.Tensor:
        if self.use_qjl:
            indices, norms, qjl_signs, gamma = args
            return self.prod_quantizer.dequantize(indices, norms, qjl_signs, gamma)
        else:
            indices, norms = args[0], args[1]
            return self.mse_quantizer.dequantize(indices, norms)


# ----------------------------------------------------------------------
# Drop-in compatible TurboQuantKVCache
# ----------------------------------------------------------------------
class TurboQuantKVCache:
    def __init__(self, num_layers: int, d_head: int, bits: int, device: str):
        self.num_layers = num_layers
        self.d_head = d_head
        self.bits = bits
        self.device = device

        # Asymmetric QJL: Key gets 2-stage (Prod), Value gets MSE-only
        self.key_quantizers: List[AdaptiveTurboQuantProd] = []
        self.val_quantizers: List[TurboQuantMSE] = []
        for li in range(num_layers):
            self.key_quantizers.append(
                AdaptiveTurboQuantProd(d_head, bits, device, seed=SEED + li * 100))
            self.val_quantizers.append(
                TurboQuantMSE(d_head, bits, device, seed=SEED + li * 100 + 50))

        self.cache: List[Optional[Tuple]] = [None] * num_layers

    def store(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        k_quant = self.key_quantizers[layer_idx].quantize(key)
        v_quant = self.val_quantizers[layer_idx].quantize(value)
        self.cache[layer_idx] = (k_quant, v_quant)

    def append(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor):
        new_k_q = self.key_quantizers[layer_idx].quantize(new_key)
        new_v_q = self.val_quantizers[layer_idx].quantize(new_value)

        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (new_k_q, new_v_q)
            return

        old_k_q, old_v_q = self.cache[layer_idx]

        merged_k = tuple(torch.cat([o, n], dim=2) for o, n in zip(old_k_q, new_k_q))
        merged_v = tuple(torch.cat([o, n], dim=2) for o, n in zip(old_v_q, new_v_q))
        self.cache[layer_idx] = (merged_k, merged_v)

    def retrieve(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.cache[layer_idx]
        if entry is None:
            raise ValueError(f"No cache for layer {layer_idx}")
        k_quant, v_quant = entry
        key = self.key_quantizers[layer_idx].dequantize(*k_quant)
        val = self.val_quantizers[layer_idx].dequantize(*v_quant)
        return key, val

    def compressed_size_kb(self) -> float:
        total_bits = 0
        for entry in self.cache:
            if entry is None:
                continue
            k_quant, v_quant = entry
            
            # Key (AdaptiveTurboQuantProd)
            if len(k_quant) == 4 and k_quant[2] is not None:
                k_indices, k_norms, k_qjl_signs, k_gamma = k_quant
                total_bits += k_indices.numel() * (self.bits - 1)
                total_bits += k_qjl_signs.numel() * 1
                total_bits += k_norms.numel() * 32
                total_bits += k_gamma.numel() * 32
            else:
                k_indices, k_norms = k_quant[0], k_quant[1]
                total_bits += k_indices.numel() * self.bits
                total_bits += k_norms.numel() * 32

            # Value (TurboQuantMSE)
            v_indices, v_norms = v_quant
            total_bits += v_indices.numel() * self.bits
            total_bits += v_norms.numel() * 32
            
        return total_bits / (8 * 1024)

    def clear(self):
        self.cache = [None] * self.num_layers


# ----------------------------------------------------------------------
# Runner Functions (adapted for prompt generation)
# ----------------------------------------------------------------------
def load_model_and_tokenizer():
    print(f"Loading {MODEL_NAME} on {DEVICE} ({DTYPE}) ...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    return model, tokenizer


def compute_perplexity(model, tokenizer, text: str) -> float:
    encodings = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return math.exp(outputs.loss.item())


def run_wht_quantizer(model, tokenizer, prompt: str) -> Dict:
    torch.cuda.empty_cache()
    gc.collect()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    num_layers = model.config.n_layer
    d_head = model.config.n_embd // model.config.n_head

    # Create WHT cache
    wht_cache = TurboQuantKVCache(
        num_layers=num_layers, d_head=d_head,
        bits=QUANT_BITS, device=DEVICE
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    generated_ids = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits
        past_kv = outputs.past_key_values

        for li in range(num_layers):
            k, v = past_kv[li]
            wht_cache.store(li, k, v)

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        for step in range(MAX_NEW_TOKENS - 1):
            dequant_past = []
            for li in range(num_layers):
                dk, dv = wht_cache.retrieve(li)
                dequant_past.append((dk, dv))
            dequant_past = tuple(dequant_past)

            outputs = model(next_token, past_key_values=dequant_past, use_cache=True)
            logits = outputs.logits
            new_past = outputs.past_key_values

            for li in range(num_layers):
                new_k = new_past[li][0][:, :, -1:, :]
                new_v = new_past[li][1][:, :, -1:, :]
                wht_cache.append(li, new_k, new_v)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    elapsed = t1 - t0
    n_tokens = generated_ids.shape[1] - input_ids.shape[1]
    compressed_kb = wht_cache.compressed_size_kb()
    ppl = compute_perplexity(model, tokenizer, generated_text)

    wht_cache.clear()

    return {
        "tokens_per_sec": n_tokens / elapsed,
        "kv_cache_kb": compressed_kb,
        "perplexity": ppl,
        "generated_text": generated_text,
        "elapsed_sec": elapsed,
        "n_tokens": n_tokens
    }


def main():
    print("=" * 70)
    print("  WHT + Asymmetric QJL Quantizer Benchmark")
    print("=" * 70)

    model, tokenizer = load_model_and_tokenizer()

    prompts = [
        "The future of artificial intelligence in healthcare will",
        "Quantum computing represents a fundamental shift in",
        "The most significant challenge facing climate science today is",
        "In the field of natural language processing, transformer models have",
    ]

    # Warmup
    warmup_ids = tokenizer.encode("Hello", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model.generate(warmup_ids, max_new_tokens=5, use_cache=True)
    torch.cuda.synchronize()

    new_results = []
    for i, prompt in enumerate(prompts):
        print(f"[Prompt {i + 1}/4] {prompt[:50]}...")
        res = run_wht_quantizer(model, tokenizer, prompt)
        print(f"  Speed: {res['tokens_per_sec']:.2f} tok/s, PPL: {res['perplexity']:.2f}, Cache: {res['kv_cache_kb']:.2f} KB")
        new_results.append(res)

    def avg_metrics(results_list):
        keys = ["tokens_per_sec", "kv_cache_kb", "perplexity"]
        return {k: float(np.mean([r[k] for r in results_list])) for k in keys}

    new_avg = avg_metrics(new_results)

    # Read original results to compare side-by-side
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), "results")
    orig_path = os.path.join(results_dir, "results.json")
    
    with open(orig_path, "r", encoding="utf-8") as f:
        orig = json.load(f)

    baseline_avg = orig["baseline_avg"]
    old_turbo_avg = orig["turboquant_avg"]

    print("=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print(f"Metric              | Baseline | Old QR + Sym QJL | New WHT + Asym QJL")
    print(f"Tokens/sec          | {baseline_avg['tokens_per_sec']:.2f}     | {old_turbo_avg['tokens_per_sec']:.2f}            | {new_avg['tokens_per_sec']:.2f}")
    print(f"KV Cache (KB)       | {baseline_avg['kv_cache_kb']:.2f}  | {old_turbo_avg['kv_cache_kb']:.2f}          | {new_avg['kv_cache_kb']:.2f}")
    print(f"Perplexity          | {baseline_avg['perplexity']:.2f}      | {old_turbo_avg['perplexity']:.2f}             | {new_avg['perplexity']:.2f}")
    print()

    # Save to JSON
    out_json = {
        "baseline_avg": baseline_avg,
        "old_turbo_avg": old_turbo_avg,
        "new_wht_avg": {k: round(v, 4) for k, v in new_avg.items()},
        "per_prompt": [
            {
                "prompt": prompts[i],
                "tokens_per_sec": round(r["tokens_per_sec"], 4),
                "kv_cache_kb": round(r["kv_cache_kb"], 4),
                "perplexity": round(r["perplexity"], 4),
                "generated_text": r["generated_text"]
            }
            for i, r in enumerate(new_results)
        ]
    }

    wht_json_path = os.path.join(results_dir, "wht_results.json")
    with open(wht_json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    # Generate HTML side-by-side dashboard
    html_path = os.path.join(results_dir, "wht_results.html")
    generate_html(out_json, html_path)


def generate_html(data: Dict, output_path: str):
    baseline = data["baseline_avg"]
    old_turbo = data["old_turbo_avg"]
    new_wht = data["new_wht_avg"]

    def bar_pct(val, max_val):
        return (val / max(max_val, 1e-5)) * 100

    max_speed = max(baseline["tokens_per_sec"], old_turbo["tokens_per_sec"], new_wht["tokens_per_sec"])
    max_kv = max(baseline["kv_cache_kb"], old_turbo["kv_cache_kb"], new_wht["kv_cache_kb"])
    max_ppl = max(baseline["perplexity"], old_turbo["perplexity"], new_wht["perplexity"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TurboQuant WHT + Asymmetric QJL Comparison</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0f1117;
    color: #e6e6e6;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    padding: 40px 20px;
    min-height: 100vh;
  }}
  .container {{
    max-width: 1100px;
    margin: 0 auto;
  }}
  header {{
    text-align: center;
    margin-bottom: 48px;
  }}
  header h1 {{
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
  }}
  header .subtitle {{
    font-size: 14px;
    color: #8b8fa3;
    font-weight: 400;
  }}
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 20px;
    margin-bottom: 40px;
  }}
  .card {{
    background: #1a1d28;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 24px;
  }}
  .card h3 {{
    font-size: 13px;
    font-weight: 600;
    color: #8b8fa3;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 20px;
  }}
  .bar-group {{
    display: grid;
    grid-template-columns: 140px 1fr 90px;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
  }}
  .bar-label {{
    font-size: 12px;
    color: #a0a4b8;
    font-weight: 500;
  }}
  .bar-container {{
    height: 22px;
    background: #12141d;
    border-radius: 6px;
    overflow: hidden;
  }}
  .bar {{
    height: 100%;
    border-radius: 6px;
    min-width: 4px;
    transition: width 0.6s ease;
  }}
  .bar-value {{
    font-size: 13px;
    font-weight: 600;
    color: #e6e6e6;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  .summary {{
    background: #1a1d28;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 32px;
    margin-bottom: 24px;
  }}
  .summary h2 {{
    font-size: 16px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 24px;
  }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    text-align: center;
  }}
  .stat .value {{
    font-size: 28px;
    font-weight: 700;
    color: #50fa7b;
    display: block;
    margin-bottom: 4px;
  }}
  .stat .value.warn {{
    color: #ffb86c;
  }}
  .stat .value.neutral {{
    color: #4a9eff;
  }}
  .stat .label {{
    font-size: 12px;
    color: #8b8fa3;
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }}
  .footer {{
    text-align: center;
    margin-top: 24px;
    font-size: 12px;
    color: #555;
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>WHT + Asymmetric QJL KV Cache Benchmark</h1>
    <div class="subtitle">Replacing QR with O(d log d) Walsh-Hadamard Transform &amp; K-Only QJL &middot; GPT-2 Medium</div>
  </header>

  <div class="cards">
    <!-- Throughput Card -->
    <div class="card">
      <h3>Throughput (Tokens / sec)</h3>
      <div class="bar-group">
        <div class="bar-label">Baseline (FP16)</div>
        <div class="bar-container">
          <div class="bar" style="width:{bar_pct(baseline['tokens_per_sec'], max_speed):.1f}%;background:#4a9eff;"></div>
        </div>
        <div class="bar-value">{baseline['tokens_per_sec']:.2f} tok/s</div>
      </div>
      <div class="bar-group">
        <div class="bar-label">QR + Sym QJL (Old)</div>
        <div class="bar-container">
          <div class="bar" style="width:{bar_pct(old_turbo['tokens_per_sec'], max_speed):.1f}%;background:#ff6b6b;"></div>
        </div>
        <div class="bar-value">{old_turbo['tokens_per_sec']:.2f} tok/s</div>
      </div>
      <div class="bar-group">
        <div class="bar-label">WHT + Asym QJL (New)</div>
        <div class="bar-container">
          <div class="bar" style="width:{bar_pct(new_wht['tokens_per_sec'], max_speed):.1f}%;background:#50fa7b;"></div>
        </div>
        <div class="bar-value">{new_wht['tokens_per_sec']:.2f} tok/s</div>
      </div>
    </div>

    <!-- KV Cache Size Card -->
    <div class="card">
      <h3>KV Cache Footprint</h3>
      <div class="bar-group">
        <div class="bar-label">Baseline (FP16)</div>
        <div class="bar-container">
          <div class="bar" style="width:{bar_pct(baseline['kv_cache_kb'], max_kv):.1f}%;background:#4a9eff;"></div>
        </div>
        <div class="bar-value">{baseline['kv_cache_kb']:.2f} KB</div>
      </div>
      <div class="bar-group">
        <div class="bar-label">QR + Sym QJL (Old)</div>
        <div class="bar-container">
          <div class="bar" style="width:{bar_pct(old_turbo['kv_cache_kb'], max_kv):.1f}%;background:#ff6b6b;"></div>
        </div>
        <div class="bar-value">{old_turbo['kv_cache_kb']:.2f} KB</div>
      </div>
      <div class="bar-group">
        <div class="bar-label">WHT + Asym QJL (New)</div>
        <div class="bar-container">
          <div class="bar" style="width:{bar_pct(new_wht['kv_cache_kb'], max_kv):.1f}%;background:#50fa7b;"></div>
        </div>
        <div class="bar-value">{new_wht['kv_cache_kb']:.2f} KB</div>
      </div>
    </div>

    <!-- Perplexity Card -->
    <div class="card">
      <h3>Perplexity (Lower is Better)</h3>
      <div class="bar-group">
        <div class="bar-label">Baseline (FP16)</div>
        <div class="bar-container">
          <div class="bar" style="width:{bar_pct(baseline['perplexity'], max_ppl):.1f}%;background:#4a9eff;"></div>
        </div>
        <div class="bar-value">{baseline['perplexity']:.2f}</div>
      </div>
      <div class="bar-group">
        <div class="bar-label">QR + Sym QJL (Old)</div>
        <div class="bar-container">
          <div class="bar" style="width:{bar_pct(old_turbo['perplexity'], max_ppl):.1f}%;background:#ffb86c;"></div>
        </div>
        <div class="bar-value">{old_turbo['perplexity']:.2f}</div>
      </div>
      <div class="bar-group">
        <div class="bar-label">WHT + Asym QJL (New)</div>
        <div class="bar-container">
          <div class="bar" style="width:{bar_pct(new_wht['perplexity'], max_ppl):.1f}%;background:#50fa7b;"></div>
        </div>
        <div class="bar-value">{new_wht['perplexity']:.2f}</div>
      </div>
    </div>
  </div>

  <div class="summary">
    <h2>Summary Comparison</h2>
    <div class="summary-grid">
      <div class="stat">
        <span class="value">{baseline['kv_cache_kb']/new_wht['kv_cache_kb']:.1f}x</span>
        <span class="label">New Compression Ratio</span>
      </div>
      <div class="stat">
        <span class="value">{((new_wht['perplexity']-baseline['perplexity'])/baseline['perplexity'])*100:+.2f}%</span>
        <span class="label">New Perplexity Change</span>
      </div>
      <div class="stat">
        <span class="value">{((new_wht['tokens_per_sec']-old_turbo['tokens_per_sec'])/old_turbo['tokens_per_sec'])*100:+.1f}%</span>
        <span class="label">Speedup vs Old TurboQuant</span>
      </div>
      <div class="stat">
        <span class="value neutral">gpt2-medium</span>
        <span class="label">Model</span>
      </div>
    </div>
  </div>

  <div class="footer">
    Generated by wht_quantizer.py &middot; torch {torch.__version__} &middot; Device: {DEVICE.upper()}
  </div>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved to {output_path}")


if __name__ == "__main__":
    main()
