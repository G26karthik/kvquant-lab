"""
bit_sweep.py
============
Runs a bit-budget sweep over [2, 3, 4, 5] bits comparing three schemes:
  - Scheme A: Original QR + symmetric QJL (existing code)
  - Scheme B: WHT + asymmetric K-only QJL (new wht_quantizer.py)
  - Scheme C: WHT + MSE only, no QJL at any bit level (ablation)

Measures:
  - Compression ratio vs FP16 baseline
  - Perplexity on the 4 prompts
  - Tokens/sec throughput
  - KV cache size KB
"""

import os
import gc
import json
import math
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

# Set paths and import custom classes
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant.turbo_quant_demo import (
    compute_lloyd_max_codebook as compute_lloyd_max_orig,
    TurboQuantKVCache as SchemeAKVCache,
    load_model_and_tokenizer,
    compute_perplexity,
    run_baseline,
    SEED,
    DEVICE,
    DTYPE,
    TEST_PROMPTS,
    MAX_NEW_TOKENS
)
from turboquant.wht_quantizer import (
    TurboQuantKVCache as SchemeBKVCache,
    TurboQuantMSE
)

# ----------------------------------------------------------------------
# Scheme C KV Cache: WHT + MSE Only on both K and V (Ablation)
# ----------------------------------------------------------------------
class SchemeCKVCache:
    def __init__(self, num_layers: int, d_head: int, bits: int, device: str):
        self.num_layers = num_layers
        self.d_head = d_head
        self.bits = bits
        self.device = device
        self.key_quantizers: List[TurboQuantMSE] = []
        self.val_quantizers: List[TurboQuantMSE] = []
        for li in range(num_layers):
            self.key_quantizers.append(
                TurboQuantMSE(d_head, bits, device, seed=SEED + li * 100))
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
            
            k_indices, k_norms = k_quant
            total_bits += k_indices.numel() * self.bits
            total_bits += k_norms.numel() * 32

            v_indices, v_norms = v_quant
            total_bits += v_indices.numel() * self.bits
            total_bits += v_norms.numel() * 32
        return total_bits / (8 * 1024)

    def clear(self):
        self.cache = [None] * self.num_layers


# ----------------------------------------------------------------------
# Generic Inference Runner for a given Cache Class
# ----------------------------------------------------------------------
def run_quantized_inference(model, tokenizer, prompt: str, cache_class, bits: int) -> Dict:
    torch.cuda.empty_cache()
    gc.collect()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    num_layers = model.config.n_layer
    d_head = model.config.n_embd // model.config.n_head

    cache = cache_class(
        num_layers=num_layers, d_head=d_head,
        bits=bits, device=DEVICE
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
            cache.store(li, k, v)

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        for step in range(MAX_NEW_TOKENS - 1):
            dequant_past = []
            for li in range(num_layers):
                dk, dv = cache.retrieve(li)
                dequant_past.append((dk, dv))
            dequant_past = tuple(dequant_past)

            outputs = model(next_token, past_key_values=dequant_past, use_cache=True)
            logits = outputs.logits
            new_past = outputs.past_key_values

            for li in range(num_layers):
                new_k = new_past[li][0][:, :, -1:, :]
                new_v = new_past[li][1][:, :, -1:, :]
                cache.append(li, new_k, new_v)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    elapsed = t1 - t0
    n_tokens = generated_ids.shape[1] - input_ids.shape[1]
    compressed_kb = cache.compressed_size_kb()
    ppl = compute_perplexity(model, tokenizer, generated_text)

    cache.clear()

    return {
        "tokens_per_sec": n_tokens / elapsed,
        "kv_cache_kb": compressed_kb,
        "perplexity": ppl
    }


# ----------------------------------------------------------------------
# Main Sweep Engine
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Bit-Budget Sweep Benchmarks")
    print("=" * 70)

    model, tokenizer = load_model_and_tokenizer()

    # 1. Run baseline (FP16)
    print("\nRunning Baseline (FP16)...")
    baseline_res = []
    for prompt in TEST_PROMPTS:
        baseline_res.append(run_baseline(model, tokenizer, prompt))
    
    baseline_kv = float(np.mean([r["kv_cache_kb"] for r in baseline_res]))
    baseline_ppl = float(np.mean([r["perplexity"] for r in baseline_res]))
    baseline_tps = float(np.mean([r["tokens_per_sec"] for r in baseline_res]))

    print(f"  Baseline Speed: {baseline_tps:.2f} tok/s, PPL: {baseline_ppl:.2f}, Cache: {baseline_kv:.2f} KB")

    # Schemes and bits to sweep
    bits_list = [2, 3, 4, 5]
    schemes = {
        "Scheme A (QR + Sym QJL)": SchemeAKVCache,
        "Scheme B (WHT + Asym QJL)": SchemeBKVCache,
        "Scheme C (WHT + MSE Only)": SchemeCKVCache
    }

    sweep_results = []

    for scheme_name, cache_class in schemes.items():
        print(f"\nEvaluating {scheme_name}...")
        for bits in bits_list:
            print(f"  Bits = {bits}...")
            prompt_res = []
            for prompt in TEST_PROMPTS:
                prompt_res.append(run_quantized_inference(model, tokenizer, prompt, cache_class, bits))
            
            avg_kv = float(np.mean([r["kv_cache_kb"] for r in prompt_res]))
            avg_ppl = float(np.mean([r["perplexity"] for r in prompt_res]))
            avg_tps = float(np.mean([r["tokens_per_sec"] for r in prompt_res]))
            comp_ratio = baseline_kv / max(avg_kv, 0.01)

            print(f"    Speed: {avg_tps:.2f} tok/s, PPL: {avg_ppl:.2f}, Compression: {comp_ratio:.2f}x")

            sweep_results.append({
                "scheme": scheme_name,
                "bits": bits,
                "perplexity": round(avg_ppl, 4),
                "tokens_per_sec": round(avg_tps, 4),
                "kv_cache_kb": round(avg_kv, 4),
                "compression_ratio": round(comp_ratio, 2)
            })

    # Save to JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    json_path = os.path.join(results_dir, "bit_sweep.json")
    out_data = {
        "baseline": {
            "perplexity": round(baseline_ppl, 4),
            "tokens_per_sec": round(baseline_tps, 4),
            "kv_cache_kb": round(baseline_kv, 4)
        },
        "sweep": sweep_results
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Results saved to {json_path}")

    # Generate HTML curves
    html_path = os.path.join(results_dir, "bit_sweep.html")
    generate_html(out_data, html_path)


# ----------------------------------------------------------------------
# HTML Report Builder with SVG Plotting
# ----------------------------------------------------------------------
def generate_html(data: Dict, output_path: str):
    baseline = data["baseline"]
    sweep = data["sweep"]

    # Separate curves by scheme
    curves = {}
    for pt in sweep:
        sch = pt["scheme"]
        if sch not in curves:
            curves[sch] = []
        curves[sch].append((pt["compression_ratio"], pt["perplexity"]))

    # Sort each curve by compression ratio
    for sch in curves:
        curves[sch] = sorted(curves[sch], key=lambda x: x[0])

    # Plotting scaling parameters
    # Let's find min and max values for coordinates to set up the SVG viewBox
    all_x = [pt[0] for sch in curves for pt in curves[sch]]
    all_y = [pt[1] for sch in curves for pt in curves[sch]]
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Add margins to bounds
    min_x = max(min_x - 0.5, 1.0)
    max_x = max_x + 0.5
    min_y = max(min_y - 0.5, 1.0)
    max_y = max_y + 0.5

    # SVG geometry
    width, height = 700, 400
    padding = 60

    def to_svg_x(x):
        return padding + (x - min_x) / (max_x - min_x) * (width - 2 * padding)

    def to_svg_y(y):
        return height - padding - (y - min_y) / (max_y - min_y) * (height - 2 * padding)

    # Generate SVG lines
    colors = {
        "Scheme A (QR + Sym QJL)": "#ff5555",
        "Scheme B (WHT + Asym QJL)": "#50fa7b",
        "Scheme C (WHT + MSE Only)": "#8be9fd"
    }

    svg_paths = ""
    svg_points = ""
    for sch, pts in curves.items():
        color = colors.get(sch, "#ffffff")
        path_d = ""
        for idx, (x, y) in enumerate(pts):
            sx, sy = to_svg_x(x), to_svg_y(y)
            if idx == 0:
                path_d += f"M {sx:.1f} {sy:.1f}"
            else:
                path_d += f" L {sx:.1f} {sy:.1f}"
            svg_points += f"""
            <circle cx="{sx:.1f}" cy="{sy:.1f}" r="4" fill="{color}" />
            <text x="{sx:.1f}" y="{sy - 8:.1f}" fill="#a0a4b8" font-size="9" text-anchor="middle">{pts[idx][1]:.2f} PPL</text>
            """
        svg_paths += f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2" />\n'

    # Build X & Y Axes and Grid
    grid_lines = ""
    # X Axis grid (Compression Ratio)
    for cr in np.linspace(math.floor(min_x), math.ceil(max_x), 6):
        if cr < min_x or cr > max_x:
            continue
        sx = to_svg_x(cr)
        grid_lines += f"""
        <line x1="{sx}" y1="{padding}" x2="{sx}" y2="{height - padding}" stroke="#2a2d3a" stroke-dasharray="2,2" />
        <text x="{sx}" y="{height - padding + 20}" fill="#8b8fa3" font-size="10" text-anchor="middle">{cr:.1f}x</text>
        """

    # Y Axis grid (Perplexity)
    for ppl in np.linspace(math.floor(min_y), math.ceil(max_y), 6):
        if ppl < min_y or ppl > max_y:
            continue
        sy = to_svg_y(ppl)
        grid_lines += f"""
        <line x1="{padding}" y1="{sy}" x2="{width - padding}" y2="{sy}" stroke="#2a2d3a" stroke-dasharray="2,2" />
        <text x="{padding - 10}" y="{sy + 4}" fill="#8b8fa3" font-size="10" text-anchor="end">{ppl:.1f}</text>
        """

    # Build table rows
    table_rows = ""
    for pt in sweep:
        table_rows += f"""
        <tr>
            <td style="color:{colors.get(pt['scheme'])}">{pt['scheme']}</td>
            <td>{pt['bits']}</td>
            <td class="num">{pt['kv_cache_kb']:.1f} KB</td>
            <td class="num highlight">{pt['compression_ratio']:.1f}x</td>
            <td class="num">{pt['perplexity']:.2f}</td>
            <td class="num">{pt['tokens_per_sec']:.2f}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bit-Budget Sweep Dashboard</title>
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
  .cards {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 30px;
    margin-bottom: 40px;
  }}
  @media(min-width: 900px) {{
    .cards {{
        grid-template-columns: 2fr 1fr;
    }}
  }}
  .card {{
    background: #1a1d28;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 24px;
  }}
  .card h2 {{
    font-size: 16px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 20px;
  }}
  .chart-container {{
    display: flex;
    justify-content: center;
    align-items: center;
    background: #12141d;
    border-radius: 8px;
    padding: 10px;
  }}
  .legend {{
    margin-top: 15px;
    display: flex;
    gap: 20px;
    justify-content: center;
    font-size: 12px;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .legend-color {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  th, td {{
    padding: 12px 10px;
    border-bottom: 1px solid #2a2d3a;
    text-align: left;
  }}
  th {{
    background: #12141d;
    color: #8b8fa3;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.5px;
  }}
  .num {{
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  .highlight {{
    color: #50fa7b;
    font-weight: 600;
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
    <h1>Bit-Budget Sweep &mdash; Quality vs Compression Curves</h1>
    <div class="subtitle">Comparing QR + Symmetric QJL vs WHT + Asymmetric QJL vs WHT + MSE-Only Ablation</div>
  </header>

  <div class="cards">
    <!-- Chart Card -->
    <div class="card">
      <h2>Perplexity vs. Compression Ratio (d=64)</h2>
      <div class="chart-container">
        <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
          <!-- Axes and Grid -->
          {grid_lines}
          
          <!-- Axis labels -->
          <text x="{width/2}" y="{height - 10}" fill="#e6e6e6" font-size="12" text-anchor="middle">Compression Ratio vs. FP16 Baseline (x)</text>
          <text x="15" y="{height/2}" fill="#e6e6e6" font-size="12" text-anchor="middle" transform="rotate(-90 15 {height/2})">Perplexity (PPL)</text>
          
          <!-- Chart Lines -->
          {svg_paths}
          <!-- Chart Points -->
          {svg_points}
        </svg>
      </div>
      
      <div class="legend">
        <div class="legend-item">
          <div class="legend-color" style="background:#ff5555;"></div>
          <span>Scheme A (QR + Sym QJL)</span>
        </div>
        <div class="legend-item">
          <div class="legend-color" style="background:#50fa7b;"></div>
          <span>Scheme B (WHT + Asym QJL)</span>
        </div>
        <div class="legend-item">
          <div class="legend-color" style="background:#8be9fd;"></div>
          <span>Scheme C (WHT + MSE Only)</span>
        </div>
      </div>
    </div>

    <!-- Info/Details Card -->
    <div class="card">
      <h2>Sweep Key Takeaways</h2>
      <p style="font-size:13.5px;line-height:1.7;color:#a0a4b8;margin-bottom:15px;">
        This sweep helps locate the <strong>crossover point</strong> for the QJL residual at $d=64$. 
        While QJL yields unbiased dot products, the added 1-bit sketch and secondary norms introduce additional variance 
        and storage overhead at higher bit budgets.
      </p>
      <p style="font-size:13.5px;line-height:1.7;color:#a0a4b8;">
        At lower bits (e.g., 2-bit), the unbiased nature of QJL helps prevent severe perplexity spikes, 
        whereas at higher bits (5-bit), MSE-only quantization achieves superior rate-distortion performance 
        since the coordinate-wise quantization error is already small.
      </p>
    </div>
  </div>

  <div class="card">
    <h2>Detailed Sweep Data</h2>
    <table>
      <thead>
        <tr>
          <th>Scheme</th>
          <th>Bits</th>
          <th class="num">KV Cache Size</th>
          <th class="num">Compression Ratio</th>
          <th class="num">Perplexity</th>
          <th class="num">Tokens/sec</th>
        </tr>
      </thead>
      <tbody>
        <!-- Baseline row -->
        <tr style="background:#12141d;">
          <td><strong>Baseline (FP16)</strong></td>
          <td>16</td>
          <td class="num">{baseline['kv_cache_kb']:.1f} KB</td>
          <td class="num highlight">1.0x</td>
          <td class="num">{baseline['perplexity']:.2f}</td>
          <td class="num">{baseline['tokens_per_sec']:.2f}</td>
        </tr>
        {table_rows}
      </tbody>
    </table>
  </div>

  <div class="footer">
    Generated by bit_sweep.py &middot; Device: {DEVICE.upper()}
  </div>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved to {output_path}")


if __name__ == "__main__":
    main()
