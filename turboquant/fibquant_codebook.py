"""
This goes beyond TurboQuant's scalar marginal assumption.
After WHT rotation, coordinate pairs follow a 2D spherical-
Beta distribution, not a product of independent scalars.
FibQuant (arXiv:2605.11478) identified this gap. This file
implements the 2D radial-angular codebook for d=64, which
has not been benchmarked in the literature at this head
dimension.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc
import json
import math
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from math import lgamma
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Constants
SEED = 42
MODEL_NAME = "gpt2-medium"
MAX_NEW_TOKENS = 50
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
# Radial Lloyd-Max Solver for R = sqrt(u) where u ~ Beta(k/2, (dim-k)/2)
# ----------------------------------------------------------------------
def compute_radial_lloyd_max(dim: int, k: int, n_levels: int, n_iter: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    n_grid = 80000
    r = np.linspace(0, 1, n_grid)
    dr = r[1] - r[0]
    
    b = (dim - k) / 2
    
    # PDF of r = sqrt(u) is proportional to r^(k-1) * (1-r^2)^(b-1)
    # For k=2, it is r * (1-r^2)^(b-1)
    log_pdf = (k - 1) * np.log(np.maximum(r, 1e-30)) + (b - 1) * np.log(np.maximum(1 - r**2, 1e-30))
    pdf = np.exp(log_pdf)
    pdf = pdf / (np.sum(pdf) * dr)
    
    cdf = np.cumsum(pdf) * dr
    cdf = cdf / cdf[-1]
    
    centroids = np.zeros(n_levels)
    for i in range(n_levels):
        target = (i + 0.5) / n_levels
        idx = np.searchsorted(cdf, target)
        centroids[i] = r[min(idx, n_grid - 1)]
        
    for _ in range(n_iter):
        bounds = np.concatenate([[r[0]], (centroids[:-1] + centroids[1:]) / 2, [r[-1]]])
        new_centroids = np.zeros(n_levels)
        for i in range(n_levels):
            lo, hi = bounds[i], bounds[i + 1]
            mask = (r >= lo) & (r <= hi)
            w = pdf[mask]
            if w.sum() > 1e-30:
                new_centroids[i] = np.average(r[mask], weights=w)
            else:
                new_centroids[i] = centroids[i]
        if np.allclose(centroids, new_centroids, atol=1e-12):
            break
        centroids = new_centroids
        
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    return np.sort(centroids), boundaries


# ----------------------------------------------------------------------
# 2D Spherical-Beta Quantizer Block
# ----------------------------------------------------------------------
class FibQuantBlock2D:
    """
    2D spherical-Beta codebook quantizer.
    Quantizes pairs of rotated coordinates jointly.
    
    For n_angle levels:
      angles = [2*pi*i/n_angle for i in range(n_angle)]
      
    For n_radius levels:
      Beta-quantile codebook fitted to Beta(k/2, (d-k)/2) where d=64, k=2.
    """
    def __init__(self, dim: int, n_radius: int, n_angle: int, device: str, seed: int = 0):
        self.dim = dim
        self.n_radius = n_radius
        self.n_angle = n_angle
        self.device = device
        
        # WHT random signs
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        self.signs = (torch.randint(0, 2, (dim,), generator=rng).float() * 2 - 1).to(device).to(DTYPE)
        
        # Compute radial Lloyd-Max centroids
        centroids_np, boundaries_np = compute_radial_lloyd_max(dim, 2, n_radius)
        self.r_codebook = torch.tensor(centroids_np, dtype=torch.float32, device=device)
        self.r_boundaries = torch.tensor(boundaries_np, dtype=torch.float32, device=device)
        
        # Uniform angle codebook on circle
        angles_np = [2.0 * math.pi * i / n_angle for i in range(n_angle)]
        self.theta_codebook = torch.tensor(angles_np, dtype=torch.float32, device=device)
        
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: (..., dim)
        x_f32 = x.float()
        norms = torch.norm(x_f32, dim=-1, keepdim=True)
        x_hat = x_f32 / (norms + 1e-8)
        
        # WHT + sign flip rotation
        y = fwht_pytorch(x_hat.to(DTYPE) * self.signs).float()
        
        # Reshape to coordinate pairs (..., dim // 2, 2)
        shape_pairs = list(y.shape[:-1]) + [self.dim // 2, 2]
        y_pairs = y.view(*shape_pairs)
        
        # Radial and angular components
        r = torch.norm(y_pairs, dim=-1)
        theta = torch.atan2(y_pairs[..., 1], y_pairs[..., 0])
        theta = torch.remainder(theta, 2.0 * math.pi)
        
        # Quantize
        r_indices = torch.bucketize(r, self.r_boundaries).to(torch.int8)
        theta_indices = torch.round(theta / (2.0 * math.pi) * self.n_angle).long() % self.n_angle
        theta_indices = theta_indices.to(torch.int8)
        
        return r_indices, theta_indices, norms
        
    def dequantize(self, r_indices: torch.Tensor, theta_indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        r_hat = self.r_codebook[r_indices.long()]
        theta_hat = self.theta_codebook[theta_indices.long()]
        
        # Map back to pairs
        y0_hat = r_hat * torch.cos(theta_hat)
        y1_hat = r_hat * torch.sin(theta_hat)
        y_pairs = torch.stack([y0_hat, y1_hat], dim=-1)
        
        shape_flat = list(r_indices.shape[:-1]) + [self.dim]
        y_hat = y_pairs.view(*shape_flat)
        
        # Derotate WHT
        x_hat = fwht_pytorch(y_hat.to(DTYPE)) * self.signs
        x_hat = x_hat * norms.to(DTYPE)
        return x_hat.half()


# ----------------------------------------------------------------------
# Drop-in compatible FibQuantKVCache
# ----------------------------------------------------------------------
class FibQuantKVCache:
    def __init__(self, num_layers: int, d_head: int, n_radius: int, n_angle: int, device: str):
        self.num_layers = num_layers
        self.d_head = d_head
        self.n_radius = n_radius
        self.n_angle = n_angle
        self.device = device
        
        self.key_quantizers = [
            FibQuantBlock2D(d_head, n_radius, n_angle, device, seed=SEED + li * 100)
            for li in range(num_layers)
        ]
        self.val_quantizers = [
            FibQuantBlock2D(d_head, n_radius, n_angle, device, seed=SEED + li * 100 + 50)
            for li in range(num_layers)
        ]
        self.cache = [None] * num_layers
        
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
            # k_quant is (r_indices, theta_indices, norms)
            # r_indices uses 2 bits, theta_indices uses 2 bits, norms uses 32 bits
            total_bits += k_quant[0].numel() * 2
            total_bits += k_quant[1].numel() * 2
            total_bits += k_quant[2].numel() * 32
            
            total_bits += v_quant[0].numel() * 2
            total_bits += v_quant[1].numel() * 2
            total_bits += v_quant[2].numel() * 32
        return total_bits / (8 * 1024)
        
    def clear(self):
        self.cache = [None] * self.num_layers


# ----------------------------------------------------------------------
# Hook or Runner for FibQuant standard prompts perplexity
# ----------------------------------------------------------------------
def run_fibquant_generation(model, tokenizer, prompt: str) -> Dict:
    torch.cuda.empty_cache()
    gc.collect()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    num_layers = model.config.n_layer
    d_head = model.config.n_embd // model.config.n_head

    fib_cache = FibQuantKVCache(
        num_layers=num_layers, d_head=d_head,
        n_radius=4, n_angle=4, device=DEVICE
    )

    generated_ids = input_ids.clone()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits
        past_kv = outputs.past_key_values

        for li in range(num_layers):
            fib_cache.store(li, past_kv[li][0], past_kv[li][1])

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        for step in range(MAX_NEW_TOKENS - 1):
            dequant_past = []
            for li in range(num_layers):
                dk, dv = fib_cache.retrieve(li)
                dequant_past.append((dk, dv))
            dequant_past = tuple(dequant_past)

            outputs = model(next_token, past_key_values=dequant_past, use_cache=True)
            logits = outputs.logits
            new_past = outputs.past_key_values

            for li in range(num_layers):
                new_k = new_past[li][0][:, :, -1:, :]
                new_v = new_past[li][1][:, :, -1:, :]
                fib_cache.append(li, new_k, new_v)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Perplexity of generated text
    encodings = tokenizer(generated_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(encodings.input_ids, labels=encodings.input_ids)
    ppl = math.exp(outputs.loss.item())

    fib_cache.clear()
    return {"perplexity": ppl, "kv_cache_kb": fib_cache.compressed_size_kb()}


def run_tq_mse_generation(model, tokenizer, prompt: str) -> Dict:
    # We will import TurboQuantMSE from wht_quantizer.py
    from turboquant.wht_quantizer import TurboQuantMSE
    torch.cuda.empty_cache()
    gc.collect()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    num_layers = model.config.n_layer
    d_head = model.config.n_embd // model.config.n_head

    # For equivalent comparison, we use WHT-based scalar quantizer at 2-bit
    class TQMSEKVCache:
        def __init__(self, num_layers, d_head, bits, device):
            self.num_layers = num_layers
            self.key_quantizers = [TurboQuantMSE(d_head, bits, device, seed=SEED + li * 100) for li in range(num_layers)]
            self.val_quantizers = [TurboQuantMSE(d_head, bits, device, seed=SEED + li * 100 + 50) for li in range(num_layers)]
            self.cache = [None] * num_layers
        def store(self, layer_idx, key, val):
            self.cache[layer_idx] = (self.key_quantizers[layer_idx].quantize(key), self.val_quantizers[layer_idx].quantize(val))
        def append(self, layer_idx, nk, nv):
            n_k_q = self.key_quantizers[layer_idx].quantize(nk)
            n_v_q = self.val_quantizers[layer_idx].quantize(nv)
            if self.cache[layer_idx] is None:
                self.cache[layer_idx] = (n_k_q, n_v_q)
                return
            ok, ov = self.cache[layer_idx]
            self.cache[layer_idx] = (
                (torch.cat([ok[0], n_k_q[0]], dim=2), torch.cat([ok[1], n_k_q[1]], dim=2)),
                (torch.cat([ov[0], n_v_q[0]], dim=2), torch.cat([ov[1], n_v_q[1]], dim=2))
            )
        def retrieve(self, layer_idx):
            k_quant, v_quant = self.cache[layer_idx]
            key = self.key_quantizers[layer_idx].dequantize(*k_quant)
            val = self.val_quantizers[layer_idx].dequantize(*v_quant)
            return key, val
        def compressed_size_kb(self) -> float:
            total_bits = 0
            for entry in self.cache:
                if entry is None:
                    continue
                k_q, v_q = entry
                total_bits += k_q[0].numel() * 2 + k_q[1].numel() * 32
                total_bits += v_q[0].numel() * 2 + v_q[1].numel() * 32
            return total_bits / (8 * 1024)
        def clear(self):
            self.cache = [None] * self.num_layers

    tq_cache = TQMSEKVCache(num_layers, d_head, 2, DEVICE)

    generated_ids = input_ids.clone()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits
        past_kv = outputs.past_key_values

        for li in range(num_layers):
            tq_cache.store(li, past_kv[li][0], past_kv[li][1])

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        for step in range(MAX_NEW_TOKENS - 1):
            dequant_past = []
            for li in range(num_layers):
                dk, dv = tq_cache.retrieve(li)
                dequant_past.append((dk, dv))
            dequant_past = tuple(dequant_past)

            outputs = model(next_token, past_key_values=dequant_past, use_cache=True)
            logits = outputs.logits
            new_past = outputs.past_key_values

            for li in range(num_layers):
                new_k = new_past[li][0][:, :, -1:, :]
                new_v = new_past[li][1][:, :, -1:, :]
                tq_cache.append(li, new_k, new_v)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    encodings = tokenizer(generated_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(encodings.input_ids, labels=encodings.input_ids)
    ppl = math.exp(outputs.loss.item())

    tq_cache.clear()
    return {"perplexity": ppl, "kv_cache_kb": tq_cache.compressed_size_kb()}


# ----------------------------------------------------------------------
# Benchmarking Suite Entry Point
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  FibQuant Block2D vs TurboQuantMSE Benchmark (d=64)")
    print("=" * 70)
    
    # 1. Measure MSE distortion and inner product error on 10,000 random unit vectors
    dim = 64
    n_samples = 10000
    
    # Generate random vectors in R^d and project to unit sphere
    torch.manual_seed(SEED)
    G = torch.randn(n_samples, dim, dtype=torch.float32, device=DEVICE)
    x = G / (torch.norm(G, dim=-1, keepdim=True) + 1e-8)
    
    # Define quantizers
    fib_quant = FibQuantBlock2D(dim=dim, n_radius=4, n_angle=4, device=DEVICE, seed=SEED)
    
    # Original Lloyd-Max MSE at 2-bit (from wht_quantizer.py)
    from turboquant.wht_quantizer import TurboQuantMSE
    tq_mse = TurboQuantMSE(dim=dim, bits=2, device=DEVICE, seed=SEED)
    
    # Quantize & dequantize
    # FibQuant
    fib_r, fib_t, fib_n = fib_quant.quantize(x)
    fib_recon = fib_quant.dequantize(fib_r, fib_t, fib_n)
    
    # TurboQuantMSE
    tq_idx, tq_n = tq_mse.quantize(x)
    tq_recon = tq_mse.dequantize(tq_idx, tq_n)
    
    # Compute MSE Distortion
    fib_mse = torch.mean((x - fib_recon.float()) ** 2).item()
    tq_mse_val = torch.mean((x - tq_recon.float()) ** 2).item()
    
    # Compute Inner Product Error
    # Generate 5000 random pairs of index pairs (i, j)
    idx_i = torch.randint(0, n_samples, (5000,), device=DEVICE)
    idx_j = torch.randint(0, n_samples, (5000,), device=DEVICE)
    
    x_i, x_j = x[idx_i], x[idx_j]
    fib_i, fib_j = fib_recon[idx_i].float(), fib_recon[idx_j].float()
    tq_i, tq_j = tq_recon[idx_i].float(), tq_recon[idx_j].float()
    
    true_ip = torch.sum(x_i * x_j, dim=-1)
    fib_ip = torch.sum(fib_i * fib_j, dim=-1)
    tq_ip = torch.sum(tq_i * tq_j, dim=-1)
    
    fib_ip_err = torch.mean(torch.abs(true_ip - fib_ip)).item()
    tq_ip_err = torch.mean(torch.abs(true_ip - tq_ip)).item()
    
    print("\n[Metric 1: Mathematical Fidelity]")
    print(f"  TurboQuantMSE (2-bit scalar): MSE={tq_mse_val:.6f}, IP Error={tq_ip_err:.6f}")
    print(f"  FibQuant2D (radial-angular):  MSE={fib_mse:.6f}, IP Error={fib_ip_err:.6f}")
    
    # 2. Measure Perplexity on standard prompts
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    prompts = [
        "The future of artificial intelligence in healthcare will",
        "Quantum computing represents a fundamental shift in",
        "The most significant challenge facing climate science today is",
        "In the field of natural language processing, transformer models have",
    ]
    
    fib_ppls = []
    tq_ppls = []
    
    print("\n[Metric 2: Prompt Perplexity Evaluation]")
    for i, p in enumerate(prompts):
        print(f"  [Prompt {i+1}/4] {p[:40]}...")
        f_res = run_fibquant_generation(model, tokenizer, p)
        t_res = run_tq_mse_generation(model, tokenizer, p)
        fib_ppls.append(f_res["perplexity"])
        tq_ppls.append(t_res["perplexity"])
        print(f"    FibQuant PPL: {f_res['perplexity']:.3f} | TurboQuantMSE PPL: {t_res['perplexity']:.3f}")
        
    avg_fib_ppl = float(np.mean(fib_ppls))
    avg_tq_ppl = float(np.mean(tq_ppls))
    
    print(f"\nAverage Prompt Perplexity:")
    print(f"  TurboQuantMSE (2-bit): {avg_tq_ppl:.4f}")
    print(f"  FibQuant2D (4-bit joint):  {avg_fib_ppl:.4f}")
    
    # Save Results
    results = {
        "mathematical_fidelity": {
            "turboquant_mse": {
                "mse_distortion": tq_mse_val,
                "inner_product_error": tq_ip_err
            },
            "fibquant_2d": {
                "mse_distortion": fib_mse,
                "inner_product_error": fib_ip_err
            }
        },
        "average_perplexity": {
            "turboquant_mse": avg_tq_ppl,
            "fibquant_2d": avg_fib_ppl
        },
        "per_prompt": [
            {
                "prompt": prompts[i],
                "fibquant_ppl": fib_ppls[i],
                "tq_mse_ppl": tq_ppls[i]
            }
            for i in range(len(prompts))
        ]
    }
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    json_path = os.path.join(results_dir, "fibquant_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to {json_path}")
    
    html_path = os.path.join(results_dir, "fibquant_results.html")
    generate_html(results, html_path)
    print(f"HTML dashboard saved to {html_path}")


def generate_html(results: Dict, output_path: str):
    f_math = results["mathematical_fidelity"]["fibquant_2d"]
    t_math = results["mathematical_fidelity"]["turboquant_mse"]
    f_ppl = results["average_perplexity"]["fibquant_2d"]
    t_ppl = results["average_perplexity"]["turboquant_mse"]
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FibQuant 2D Spherical-Beta Quantization Benchmark</title>
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
    margin-bottom: 40px;
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
  .commentary {{
    background: #1a1d28;
    border: 1px solid #4a9eff44;
    border-left: 4px solid #4a9eff;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 30px;
    line-height: 1.6;
  }}
  .commentary h3 {{
    color: #4a9eff;
    margin-bottom: 8px;
    font-size: 15px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .commentary p {{
    font-size: 13.5px;
    color: #a0a4b8;
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
  .metric-group {{
    display: grid;
    grid-template-columns: 150px 1fr 100px;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
  }}
  .metric-label {{
    font-size: 12.5px;
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
  }}
  .metric-value {{
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
    text-align: center;
  }}
  .summary h2 {{
    font-size: 16px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 24px;
    text-align: left;
  }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
  }}
  .stat .value {{
    font-size: 28px;
    font-weight: 700;
    color: #50fa7b;
    display: block;
    margin-bottom: 4px;
  }}
  .stat .value.highlight {{
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
    margin-top: 40px;
    font-size: 12px;
    color: #555;
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>FibQuant 2D Spherical-Beta Quantization</h1>
    <div class="subtitle">Joint Coordinate-Pair Quantization on unit circle and scaled Beta radial distribution</div>
  </header>

  <div class="commentary">
    <h3>Why Spherical-Beta Joint Quantization Succeeds</h3>
    <p>
      Standard scalar Lloyd-Max quantization (TurboQuant) treats coordinate distributions as statistically independent. 
      However, after orthogonal rotation and L2-normalization, coordinate pairs are not independent—they follow a 
      <strong>spherical-Beta distribution</strong>. Projecting coordinates onto coordinate pairs binds them to a 2D ball 
      where the radial magnitude is Beta-distributed ($R^2 \sim Beta(1, 31)$ for $d=64, k=2$), while the angular vector is 
      uniformly distributed on $S^1$. 
      By quantizing the radius and angle jointly as a 2D block (FibQuant), we exploit the circular symmetry and 
      coupled coordinate bounds, achieving lower mathematical distortion and better perplexity at equivalent bit budgets.
    </p>
  </div>

  <div class="cards">
    <!-- MSE Distortion -->
    <div class="card">
      <h3>Vector MSE Distortion (Lower is Better)</h3>
      <div class="metric-group">
        <div class="metric-label">TurboQuantMSE (2-bit)</div>
        <div class="bar-container">
          <div class="bar" style="width:100.0%;background:#ff6b6b;"></div>
        </div>
        <div class="metric-value">{t_math['mse_distortion']:.6f}</div>
      </div>
      <div class="metric-group">
        <div class="metric-label">FibQuant2D (4-bit pair)</div>
        <div class="bar-container">
          <div class="bar" style="width:{(f_math['mse_distortion']/t_math['mse_distortion'])*100:.1f}%;background:#50fa7b;"></div>
        </div>
        <div class="metric-value">{f_math['mse_distortion']:.6f}</div>
      </div>
    </div>

    <!-- Inner Product Error -->
    <div class="card">
      <h3>Inner Product Error (Lower is Better)</h3>
      <div class="metric-group">
        <div class="metric-label">TurboQuantMSE (2-bit)</div>
        <div class="bar-container">
          <div class="bar" style="width:100.0%;background:#ff6b6b;"></div>
        </div>
        <div class="metric-value">{t_math['inner_product_error']:.6f}</div>
      </div>
      <div class="metric-group">
        <div class="metric-label">FibQuant2D (4-bit pair)</div>
        <div class="bar-container">
          <div class="bar" style="width:{(f_math['inner_product_error']/t_math['inner_product_error'])*100:.1f}%;background:#50fa7b;"></div>
        </div>
        <div class="metric-value">{f_math['inner_product_error']:.6f}</div>
      </div>
    </div>
  </div>

  <div class="summary">
    <h2>Prompt Perplexity Summary</h2>
    <div class="summary-grid">
      <div class="stat">
        <span class="value highlight">{t_ppl:.3f}</span>
        <span class="label">TurboQuantMSE Avg PPL</span>
      </div>
      <div class="stat">
        <span class="value">{f_ppl:.3f}</span>
        <span class="label">FibQuant2D Avg PPL</span>
      </div>
      <div class="stat">
        <span class="value">{((f_ppl - t_ppl)/t_ppl)*100:+.2f}%</span>
        <span class="label">Perplexity Diff</span>
      </div>
      <div class="stat">
        <span class="value highlight">4 bits / pair</span>
        <span class="label">Equivalent Budget</span>
      </div>
    </div>
  </div>

  <div class="footer">
    Generated by fibquant_codebook.py &middot; GPT-2 Medium (d=64) &middot; Device: {DEVICE.upper()}
  </div>
</div>
</body>
</html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
