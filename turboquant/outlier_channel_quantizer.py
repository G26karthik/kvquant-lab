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
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from turboquant.wht_quantizer import TurboQuantMSE

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
# OutlierChannelQuantizer (Section 4.2 of paper)
# ----------------------------------------------------------------------
class OutlierChannelQuantizer:
    """
    Outlier Channel Splitting Quantizer (Section 4.2).
    Applies random orthogonal rotation, then splits channels into outliers 
    (quantized with outlier_bits) and normal channels (quantized with normal_bits).
    """
    def __init__(self, dim: int, avg_bits: float, device: str, seed: int = 0):
        self.dim = dim
        self.avg_bits = avg_bits
        self.device = device
        self.outlier_count = dim // 4  # 25% of channels
        self.normal_count = dim - self.outlier_count
        
        # Determine bit levels automatically (exact average bits)
        if math.isclose(avg_bits, 2.5):
            self.outlier_bits = 4
            self.normal_bits = 2
        elif math.isclose(avg_bits, 3.5):
            self.outlier_bits = 5
            self.normal_bits = 3
        else:
            # Fallback/general case (keeps average at approx avg_bits)
            self.outlier_bits = math.ceil(avg_bits) + 1
            self.normal_bits = math.floor(avg_bits) - 1
            if self.normal_bits < 1:
                self.normal_bits = 1
                self.outlier_bits = math.ceil(avg_bits)
            
        # Dense QR rotation matrices for outlier and normal subspaces
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        G_out = torch.randn(self.outlier_count, self.outlier_count, generator=rng, dtype=torch.float32)
        Q_out, _ = torch.linalg.qr(G_out)
        self.rotation_out = Q_out.to(device).to(DTYPE)
        
        G_norm = torch.randn(self.normal_count, self.normal_count, generator=rng, dtype=torch.float32)
        Q_norm, _ = torch.linalg.qr(G_norm)
        self.rotation_norm = Q_norm.to(device).to(DTYPE)
        
        # Instantiate base MSE quantizers to reuse codebooks and boundaries (use sub-space dimensions)
        self.outlier_quantizer = TurboQuantMSE(dim=self.outlier_count, bits=self.outlier_bits, device=device, seed=seed)
        self.normal_quantizer = TurboQuantMSE(dim=self.normal_count, bits=self.normal_quantizer_bits_helper(), device=device, seed=seed + 1)
        
        # Outlier mask (calibrated later)
        self.outlier_mask = torch.zeros(dim, dtype=torch.bool, device=device)
        self.is_calibrated = False

    def normal_quantizer_bits_helper(self):
        # Prevent 0-bit quantization if avg_bits is extremely low
        return max(1, self.normal_bits)

    def calibrate(self, calibration_vectors: torch.Tensor):
        """
        Identify top outlier channels across a calibration batch in the original channel space.
        """
        x_f32 = calibration_vectors.float()
        norms = torch.norm(x_f32, dim=-1, keepdim=True)
        x_hat = x_f32 / (norms + 1e-8)
        
        # Compute per-channel variance in the original coordinate space
        variances = torch.var(x_hat.float(), dim=0)  # shape (dim,)
        
        # Select top outlier_count channels
        _, top_indices = torch.topk(variances, self.outlier_count)
        self.outlier_mask = torch.zeros(self.dim, dtype=torch.bool, device=self.device)
        self.outlier_mask[top_indices] = True
        self.is_calibrated = True

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_f32 = x.float()
        norms = torch.norm(x_f32, dim=-1, keepdim=True)
        x_hat = x_f32 / (norms + 1e-8)
        
        if not self.is_calibrated:
            # Calibrate on the fly if not already done
            self.calibrate(x.reshape(-1, self.dim))
            
        # Split channels in the original coordinate space
        x_outlier = x_hat[..., self.outlier_mask]
        x_normal = x_hat[..., ~self.outlier_mask]
        
        # Scale/normalize each sub-vector prior to random rotation
        norm_out = torch.norm(x_outlier.float(), dim=-1, keepdim=True)
        norm_norm = torch.norm(x_normal.float(), dim=-1, keepdim=True)
        
        x_out_scaled = x_outlier.float() / (norm_out + 1e-8)
        x_norm_scaled = x_normal.float() / (norm_norm + 1e-8)
        
        # Rotate sub-vectors independently
        y_outlier = (x_out_scaled.to(DTYPE) @ self.rotation_out.T).float()
        y_normal = (x_norm_scaled.to(DTYPE) @ self.rotation_norm.T).float()
        
        # Quantize split channels using Lloyd-Max boundaries
        indices_outlier = torch.bucketize(y_outlier.contiguous(), self.outlier_quantizer.boundaries).to(torch.int8)
        indices_normal = torch.bucketize(y_normal.contiguous(), self.normal_quantizer.boundaries).to(torch.int8)
        
        return indices_outlier, indices_normal, norms, norm_out, norm_norm

    def dequantize(self, compressed: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        indices_outlier, indices_normal, norms, norm_out, norm_norm = compressed
        
        # Look up values in codebooks
        y_hat_outlier = self.outlier_quantizer.codebook[indices_outlier.long()]
        y_hat_normal = self.normal_quantizer.codebook[indices_normal.long()]
        
        # Rotate back and scale by sub-vector norms
        x_out_hat = (y_hat_outlier.to(DTYPE) @ self.rotation_out) * norm_out.to(DTYPE)
        x_norm_hat = (y_hat_normal.to(DTYPE) @ self.rotation_norm) * norm_norm.to(DTYPE)
        
        # Reconstruct coordinate vector
        x_hat = torch.zeros(*indices_outlier.shape[:-1], self.dim, dtype=DTYPE, device=self.device)
        x_hat[..., self.outlier_mask] = x_out_hat
        x_hat[..., ~self.outlier_mask] = x_norm_hat
        
        # Rescale by global norm
        x_hat = x_hat * norms.to(DTYPE)
        
        return x_hat.half()


# ----------------------------------------------------------------------
# Drop-in compatible OutlierKVCache
# ----------------------------------------------------------------------
class OutlierKVCache:
    def __init__(self, num_layers: int, d_head: int, avg_bits: float, device: str):
        self.num_layers = num_layers
        self.d_head = d_head
        self.avg_bits = avg_bits
        self.device = device
        
        self.key_quantizers = [
            OutlierChannelQuantizer(d_head, avg_bits, device, seed=SEED + li * 100)
            for li in range(num_layers)
        ]
        self.val_quantizers = [
            OutlierChannelQuantizer(d_head, avg_bits, device, seed=SEED + li * 100 + 50)
            for li in range(num_layers)
        ]
        self.cache = [None] * num_layers

    def calibrate_all(self, cal_vectors: torch.Tensor):
        for li in range(self.num_layers):
            self.key_quantizers[li].calibrate(cal_vectors)
            self.val_quantizers[li].calibrate(cal_vectors)

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
        k_quant, v_quant = self.cache[layer_idx]
        key = self.key_quantizers[layer_idx].dequantize(k_quant)
        val = self.val_quantizers[layer_idx].dequantize(v_quant)
        return key, val

    def compressed_size_kb(self) -> float:
        total_bits = 0
        for entry in self.cache:
            if entry is None:
                continue
            k_quant, v_quant = entry
            
            # Key footprint
            k_out, k_norm, norms, norm_out, norm_norm = k_quant
            total_bits += k_out.numel() * self.key_quantizers[0].outlier_bits
            total_bits += k_norm.numel() * self.key_quantizers[0].normal_bits
            total_bits += norms.numel() * 32
            total_bits += norm_out.numel() * 32
            total_bits += norm_norm.numel() * 32
            
            # Value footprint
            v_out, v_norm, norms, norm_out, norm_norm = v_quant
            total_bits += v_out.numel() * self.val_quantizers[0].outlier_bits
            total_bits += v_norm.numel() * self.val_quantizers[0].normal_bits
            total_bits += norms.numel() * 32
            total_bits += norm_out.numel() * 32
            total_bits += norm_norm.numel() * 32
            
        return total_bits / (8 * 1024)

    def clear(self):
        self.cache = [None] * self.num_layers


# ----------------------------------------------------------------------
# Flat TQMSEKVCache to run comparison on the fly
# ----------------------------------------------------------------------
class TQMSEKVCache:
    def __init__(self, num_layers: int, d_head: int, bits: int, device: str):
        self.num_layers = num_layers
        self.d_head = d_head
        self.bits = bits
        self.device = device
        self.quantizers = [
            TurboQuantMSE(d_head, bits, device, seed=SEED + li * 100)
            for li in range(num_layers)
        ]
        self.cache = [None] * num_layers

    def store(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        k_quant = self.quantizers[layer_idx].quantize(key)
        v_quant = self.quantizers[layer_idx].quantize(value)
        self.cache[layer_idx] = (k_quant, v_quant)

    def append(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor):
        new_k_q = self.quantizers[layer_idx].quantize(new_key)
        new_v_q = self.quantizers[layer_idx].quantize(new_value)

        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (new_k_q, new_v_q)
            return

        old_k_q, old_v_q = self.cache[layer_idx]
        merged_k = tuple(torch.cat([o, n], dim=2) for o, n in zip(old_k_q, new_k_q))
        merged_v = tuple(torch.cat([o, n], dim=2) for o, n in zip(old_v_q, new_v_q))
        self.cache[layer_idx] = (merged_k, merged_v)

    def retrieve(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        k_quant, v_quant = self.cache[layer_idx]
        key = self.quantizers[layer_idx].dequantize(*k_quant)
        val = self.quantizers[layer_idx].dequantize(*v_quant)
        return key, val

    def compressed_size_kb(self) -> float:
        total_bits = 0
        for entry in self.cache:
            if entry is None:
                continue
            k_quant, v_quant = entry
            total_bits += k_quant[0].numel() * self.bits + k_quant[1].numel() * 32
            total_bits += v_quant[0].numel() * self.bits + v_quant[1].numel() * 32
        return total_bits / (8 * 1024)

    def clear(self):
        self.cache = [None] * self.num_layers


# ----------------------------------------------------------------------
# Generation & Perplexity Helpers
# ----------------------------------------------------------------------
def run_generation(model, tokenizer, prompt: str, cache_obj) -> Dict:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    num_layers = model.config.n_layer
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits
        past_kv = outputs.past_key_values
        
        for li in range(num_layers):
            cache_obj.store(li, past_kv[li][0], past_kv[li][1])
            
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        for step in range(MAX_NEW_TOKENS - 1):
            dequant_past = []
            for li in range(num_layers):
                dk, dv = cache_obj.retrieve(li)
                dequant_past.append((dk, dv))
            dequant_past = tuple(dequant_past)
            
            outputs = model(next_token, past_key_values=dequant_past, use_cache=True)
            logits = outputs.logits
            new_past = outputs.past_key_values
            
            for li in range(num_layers):
                new_k = new_past[li][0][:, :, -1:, :]
                new_v = new_past[li][1][:, :, -1:, :]
                cache_obj.append(li, new_k, new_v)
                
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    encodings = tokenizer(gen_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(encodings.input_ids, labels=encodings.input_ids)
    ppl = math.exp(outputs.loss.item())
    
    size_kb = cache_obj.compressed_size_kb()
    cache_obj.clear()
    
    return {"perplexity": ppl, "kv_cache_kb": size_kb}


# ----------------------------------------------------------------------
# Main Benchmarking Suite
# ----------------------------------------------------------------------
def main():
    print("=" * 75)
    print("  TurboQuant Outlier Channel Splitting Evaluation (d=64)")
    print("=" * 75)
    
    # 1. Calibration
    dim = 64
    n_cal = 1000
    torch.manual_seed(SEED)
    G_cal = torch.randn(n_cal, dim, dtype=torch.float32, device=DEVICE)
    cal_vectors = G_cal / (torch.norm(G_cal, dim=-1, keepdim=True) + 1e-8)
    
    # Define schemes
    schemes = {
        "outlier_2.5": OutlierChannelQuantizer(dim=dim, avg_bits=2.5, device=DEVICE, seed=SEED),
        "outlier_3.5": OutlierChannelQuantizer(dim=dim, avg_bits=3.5, device=DEVICE, seed=SEED),
        "flat_3": TurboQuantMSE(dim=dim, bits=3, device=DEVICE, seed=SEED),
        "flat_4": TurboQuantMSE(dim=dim, bits=4, device=DEVICE, seed=SEED)
    }
    
    # Calibrate outliers
    schemes["outlier_2.5"].calibrate(cal_vectors)
    schemes["outlier_3.5"].calibrate(cal_vectors)
    
    # 2. Mathematical Fidelity Evaluation (10,000 vectors)
    n_samples = 10000
    G_eval = torch.randn(n_samples, dim, dtype=torch.float32, device=DEVICE)
    x = G_eval / (torch.norm(G_eval, dim=-1, keepdim=True) + 1e-8)
    
    recon = {}
    
    # Outlier 2.5
    o25_c = schemes["outlier_2.5"].quantize(x)
    recon["outlier_2.5"] = schemes["outlier_2.5"].dequantize(o25_c)
    
    # Outlier 3.5
    o35_c = schemes["outlier_3.5"].quantize(x)
    recon["outlier_3.5"] = schemes["outlier_3.5"].dequantize(o35_c)
    
    # Flat 3
    f3_c = schemes["flat_3"].quantize(x)
    recon["flat_3"] = schemes["flat_3"].dequantize(*f3_c)
    
    # Flat 4
    f4_c = schemes["flat_4"].quantize(x)
    recon["flat_4"] = schemes["flat_4"].dequantize(*f4_c)
    
    # Compute MSE Distortion
    mse_dist = {}
    for name, rx in recon.items():
        mse_dist[name] = torch.mean((x - rx.float()) ** 2).item()
        
    # Compute Inner Product Error (5,000 random pairs)
    idx_i = torch.randint(0, n_samples, (5000,), device=DEVICE)
    idx_j = torch.randint(0, n_samples, (5000,), device=DEVICE)
    x_i, x_j = x[idx_i], x[idx_j]
    true_ip = torch.sum(x_i * x_j, dim=-1)
    
    ip_error = {}
    for name, rx in recon.items():
        rx_i, rx_j = rx[idx_i].float(), rx[idx_j].float()
        approx_ip = torch.sum(rx_i * rx_j, dim=-1)
        ip_error[name] = torch.mean(torch.abs(true_ip - approx_ip)).item()
        
    print("\n[Metric 1: Mathematical Fidelity]")
    for name in mse_dist:
        print(f"  {name:12s}: MSE={mse_dist[name]:.6f}, Inner Product Error={ip_error[name]:.6f}")
        
    # 3. Prompt Perplexity Evaluation
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
    
    num_layers = model.config.n_layer
    d_head = model.config.n_embd // model.config.n_head
    
    # Caches
    caches = {
        "outlier_2.5": OutlierKVCache(num_layers, d_head, 2.5, DEVICE),
        "outlier_3.5": OutlierKVCache(num_layers, d_head, 3.5, DEVICE),
        "flat_3": TQMSEKVCache(num_layers, d_head, 3, DEVICE),
        "flat_4": TQMSEKVCache(num_layers, d_head, 4, DEVICE)
    }
    
    # Calibrate caches
    caches["outlier_2.5"].calibrate_all(cal_vectors)
    caches["outlier_3.5"].calibrate_all(cal_vectors)
    
    prompt_results = {name: [] for name in caches}
    
    print("\n[Metric 2: Prompt Perplexity Evaluation]")
    for p_idx, prompt in enumerate(prompts):
        print(f"  [Prompt {p_idx+1}/4] {prompt[:40]}...")
        for name, cache_obj in caches.items():
            res = run_generation(model, tokenizer, prompt, cache_obj)
            prompt_results[name].append(res)
            print(f"    {name:12s} -> PPL: {res['perplexity']:.3f} | Cache: {res['kv_cache_kb']:.2f} KB")
            
    # Compute Averages
    avg_ppl = {}
    avg_size = {}
    for name in caches:
        avg_ppl[name] = float(np.mean([r["perplexity"] for r in prompt_results[name]]))
        avg_size[name] = float(np.mean([r["kv_cache_kb"] for r in prompt_results[name]]))
        
    print("\n[Averages]")
    for name in caches:
        print(f"  {name:12s}: PPL={avg_ppl[name]:.4f}, Cache Size={avg_size[name]:.2f} KB")
        
    # Save to JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    out_json = {
        "mathematical_fidelity": {
            name: {
                "mse_distortion": mse_dist[name],
                "inner_product_error": ip_error[name]
            }
            for name in mse_dist
        },
        "average_perplexity": avg_ppl,
        "average_cache_size_kb": avg_size,
        "per_prompt": [
            {
                "prompt": prompts[i],
                "results": {
                    name: {
                        "perplexity": prompt_results[name][i]["perplexity"],
                        "kv_cache_kb": prompt_results[name][i]["kv_cache_kb"]
                    }
                    for name in caches
                }
            }
            for i in range(len(prompts))
        ]
    }
    
    json_path = os.path.join(results_dir, "outlier_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nJSON results saved to {json_path}")
    
    # Generate HTML
    html_path = os.path.join(results_dir, "outlier_results.html")
    generate_html(out_json, html_path)
    print(f"HTML report saved to {html_path}")


def generate_html(data: Dict, output_path: str):
    fidelity = data["mathematical_fidelity"]
    avg_ppl = data["average_perplexity"]
    avg_size = data["average_cache_size_kb"]
    
    rows = ""
    for name in fidelity:
        label = name.replace("_", " ").title()
        rows += f"""
        <tr>
            <td class="dim-val">{label}</td>
            <td class="num-val">{fidelity[name]['mse_distortion']:.6f}</td>
            <td class="num-val">{fidelity[name]['inner_product_error']:.6f}</td>
            <td class="num-val highlight">{avg_ppl[name]:.4f}</td>
            <td class="num-val">{avg_size[name]:.2f} KB</td>
        </tr>
        """
        
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TurboQuant Outlier Channel Splitting Benchmark</title>
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
    max-width: 1000px;
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
  .card {{
    background: #1a1d28;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 30px;
  }}
  .card h2 {{
    font-size: 16px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 20px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    font-size: 13px;
  }}
  th, td {{
    padding: 12px 15px;
    text-align: left;
    border-bottom: 1px solid #2a2d3a;
  }}
  th {{
    background: #12141d;
    color: #8b8fa3;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
  }}
  tr:hover {{
    background: #242936;
  }}
  .num-val {{
    font-variant-numeric: tabular-nums;
    text-align: right;
  }}
  th.num-val {{
    text-align: right;
  }}
  .highlight {{
    color: #50fa7b;
    font-weight: 600;
  }}
  .dim-val {{
    font-weight: 600;
    color: #4a9eff;
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
    <h1>Outlier Channel Splitting Evaluation Dashboard</h1>
    <div class="subtitle">Comparing Outlier Splitting (2.5-bit and 3.5-bit) vs. Flat Lloyd-Max (3-bit and 4-bit) &middot; GPT-2 Medium</div>
  </header>

  <div class="card">
    <h2>Outlier Channel Splitting Benchmark Results</h2>
    <table>
      <thead>
        <tr>
          <th>Configuration</th>
          <th class="num-val">MSE Distortion</th>
          <th class="num-val">Inner Product Error</th>
          <th class="num-val">Avg Perplexity</th>
          <th class="num-val">Avg KV Cache Footprint</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>

  <div class="footer">
    Generated by outlier_channel_quantizer.py &middot; GPT-2 Medium &middot; Device: {DEVICE.upper()}
  </div>
</div>
</body>
</html>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
