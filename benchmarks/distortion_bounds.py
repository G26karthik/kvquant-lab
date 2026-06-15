import os
import sys
import json
import math
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant.turbo_quant_demo import TurboQuantMSE as OrigTurboQuantMSE
from turboquant.turbo_quant_demo import TurboQuantProd as OrigTurboQuantProd
from turboquant.wht_quantizer import TurboQuantMSE as WhtTurboQuantMSE
from turboquant.wht_quantizer import AdaptiveTurboQuantProd
from turboquant.outlier_channel_quantizer import OutlierChannelQuantizer

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

torch.manual_seed(SEED)
np.random.seed(SEED)

def generate_unit_vectors(num, dim):
    x = torch.randn(num, dim, device=DEVICE)
    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    return x

def main():
    print("=" * 80)
    print("Running distortion bounds benchmark (Theorem 1 and Theorem 3)...")
    print("=" * 80)
    
    dim = 64
    num_mse_vectors = 50000
    num_ip_pairs = 10000
    
    # Generate vectors
    print("Generating 50,000 random unit vectors...")
    x_mse = generate_unit_vectors(num_mse_vectors, dim)
    
    print("Generating 10,000 query-key pairs...")
    x_ip = generate_unit_vectors(num_ip_pairs, dim)
    y_ip = generate_unit_vectors(num_ip_pairs, dim)
    
    bit_levels = [2, 3, 4, 5]
    results_mse = []
    results_ip = []
    
    for b in bit_levels:
        print(f"\nProcessing bit level b = {b}...")
        
        # 1. Compute Shannon Lower Bounds
        shannon_mse_lb = (1.0 / 4.0) ** b
        # E[||y||^2] = 1 since y_ip is normalized unit vectors
        shannon_ip_lb = (1.0 / dim) * ((1.0 / 4.0) ** b)
        
        # 2. Instantiate quantizers
        # Flat QR
        orig_mse_q = OrigTurboQuantMSE(dim, bits=b, device=DEVICE, seed=SEED)
        orig_prod_q = OrigTurboQuantProd(dim, bits=b, device=DEVICE, seed=SEED)
        
        # WHT + Asymmetric
        wht_mse_q = WhtTurboQuantMSE(dim, bits=b, device=DEVICE, seed=SEED)
        # Note: at d=64, AdaptiveTurboQuantProd disables QJL and uses MSE only.
        wht_prod_q = AdaptiveTurboQuantProd(dim, bits=b, device=DEVICE, seed=SEED)
        
        # Outlier Split (uses corrected OutlierChannelQuantizer)
        outlier_mse_q = OutlierChannelQuantizer(dim, avg_bits=float(b), device=DEVICE, seed=SEED)
        outlier_mse_q.calibrate(x_mse)
        
        # Measure actual MSE distortions
        with torch.no_grad():
            # Flat QR
            idx_orig, norm_orig = orig_mse_q.quantize(x_mse.to(DTYPE))
            recon_orig = orig_mse_q.dequantize(idx_orig, norm_orig).float()
            mse_orig = torch.mean(torch.sum((x_mse - recon_orig) ** 2, dim=-1)).item()
            
            # WHT
            idx_wht, norm_wht = wht_mse_q.quantize(x_mse.to(DTYPE))
            recon_wht = wht_mse_q.dequantize(idx_wht, norm_wht).float()
            mse_wht = torch.mean(torch.sum((x_mse - recon_wht) ** 2, dim=-1)).item()
            
            # Outlier-split
            comp_out = outlier_mse_q.quantize(x_mse.to(DTYPE))
            recon_out = outlier_mse_q.dequantize(comp_out).float()
            mse_out = torch.mean(torch.sum((x_mse - recon_out) ** 2, dim=-1)).item()
            
        # Ratios
        ratio_tq_mse = mse_orig / shannon_mse_lb
        ratio_wht_mse = mse_wht / shannon_mse_lb
        ratio_out_mse = mse_out / shannon_mse_lb
        
        print(f"  MSE - Shannon LB: {shannon_mse_lb:.6f}")
        print(f"  MSE - Flat QR (TQ): {mse_orig:.6f} (Ratio: {ratio_tq_mse:.3f})")
        print(f"  MSE - WHT:          {mse_wht:.6f} (Ratio: {ratio_wht_mse:.3f})")
        print(f"  MSE - Outlier Split: {mse_out:.6f} (Ratio: {ratio_out_mse:.3f})")
        
        # Assertions
        assert ratio_tq_mse <= 2.7, f"Flat QR MSE ratio {ratio_tq_mse:.3f} exceeds 2.7"
        assert ratio_wht_mse <= 2.7, f"WHT MSE ratio {ratio_wht_mse:.3f} exceeds 2.7"
        
        results_mse.append({
            "bits": b,
            "shannon_lb": shannon_mse_lb,
            "tq_mse": mse_orig,
            "wht_mse": mse_wht,
            "out_mse": mse_out,
            "ratio_tq": ratio_tq_mse,
            "ratio_wht": ratio_wht_mse,
            "ratio_out": ratio_out_mse
        })
        
        # Measure inner product distortions
        with torch.no_grad():
            # Flat QR (using OrigTurboQuantProd which is unbiased)
            idx_orig_ip, norm_orig_ip, qjl_orig, gamma_orig = orig_prod_q.quantize(x_ip.to(DTYPE))
            recon_orig_ip = orig_prod_q.dequantize(idx_orig_ip, norm_orig_ip, qjl_orig, gamma_orig).float()
            
            # WHT (using AdaptiveTurboQuantProd)
            wht_comp_ip = wht_prod_q.quantize(x_ip.to(DTYPE))
            recon_wht_ip = wht_prod_q.dequantize(*wht_comp_ip).float()
            
            # Compute inner products
            true_ip = torch.sum(y_ip * x_ip, dim=-1)
            ip_recon_orig = torch.sum(y_ip * recon_orig_ip, dim=-1)
            ip_recon_wht = torch.sum(y_ip * recon_wht_ip, dim=-1)
            
            # E[|<y,x> - <y,x_hat>|^2]
            ip_dist_orig = torch.mean((true_ip - ip_recon_orig) ** 2).item()
            ip_dist_wht = torch.mean((true_ip - ip_recon_wht) ** 2).item()
            
        ratio_tq_ip = ip_dist_orig / shannon_ip_lb
        ratio_wht_ip = ip_dist_wht / shannon_ip_lb
        
        print(f"  IP - Shannon LB:  {shannon_ip_lb:.8f}")
        print(f"  IP - Flat QR (TQ): {ip_dist_orig:.8f} (Ratio: {ratio_tq_ip:.3f})")
        print(f"  IP - WHT:          {ip_dist_wht:.8f} (Ratio: {ratio_wht_ip:.3f})")
        
        # Verify that our adaptive WHT inner product achieves within 2.7x of Shannon bound
        assert ratio_wht_ip <= 2.7, f"WHT IP ratio {ratio_wht_ip:.3f} exceeds 2.7"
        
        results_ip.append({
            "bits": b,
            "shannon_lb": shannon_ip_lb,
            "tq_ip": ip_dist_orig,
            "wht_ip": ip_dist_wht,
            "ratio_tq": ratio_tq_ip,
            "ratio_wht": ratio_wht_ip
        })
        
    print("\n" + "=" * 80)
    for res in results_mse:
        print(f"TurboQuant achieves {res['ratio_tq']:.2f}x the theoretical optimum at d=64 for MSE at {res['bits']}-bit")
    for res in results_ip:
        print(f"TurboQuant achieves {res['ratio_tq']:.2f}x the theoretical optimum at d=64 for IP at {res['bits']}-bit")
    print("=" * 80)
    
    # Save JSON results
    out_json = {
        "mse_results": results_mse,
        "ip_results": results_ip
    }
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "distortion_bounds.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"Saved JSON to {json_path}")
    
    # Generate HTML report
    html_path = os.path.join(results_dir, "distortion_bounds.html")
    generate_html(out_json, html_path)
    print(f"Saved HTML to {html_path}")

def generate_html(data, output_path):
    mse_rows = ""
    for r in data["mse_results"]:
        mse_rows += f"""
        <tr>
            <td class="dim-val">{r['bits']}</td>
            <td class="num-val">{r['shannon_lb']:.8f}</td>
            <td class="num-val highlight">{r['tq_mse']:.8f}</td>
            <td class="num-val">{r['wht_mse']:.8f}</td>
            <td class="num-val">{r['out_mse']:.8f}</td>
            <td class="num-val highlight">{r['ratio_tq']:.3f}x</td>
            <td class="num-val">{r['ratio_wht']:.3f}x</td>
            <td class="num-val">{r['ratio_out']:.3f}x</td>
        </tr>
        """
        
    ip_rows = ""
    for r in data["ip_results"]:
        ip_rows += f"""
        <tr>
            <td class="dim-val">{r['bits']}</td>
            <td class="num-val">{r['shannon_lb']:.8f}</td>
            <td class="num-val highlight">{r['tq_ip']:.8f}</td>
            <td class="num-val">{r['wht_ip']:.8f}</td>
            <td class="num-val highlight">{r['ratio_tq']:.3f}x</td>
            <td class="num-val">{r['ratio_wht']:.3f}x</td>
        </tr>
        """
        
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Information-Theoretic Distortion Bounds Verification</title>
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
    <h1>Information-Theoretic Shannon Distortion Bounds Verification</h1>
    <div class="subtitle">Verifying TurboQuant Distortion within factor ≈2.7 of Shannon Lower Bound &middot; d=64</div>
  </header>

  <div class="card">
    <h2>Mean Squared Error (MSE) Distortion Comparison</h2>
    <table>
      <thead>
        <tr>
          <th>Bits (b)</th>
          <th class="num-val">Shannon MSE LB</th>
          <th class="num-val">TQ MSE (Flat QR)</th>
          <th class="num-val">WHT MSE</th>
          <th class="num-val">Outlier Split MSE</th>
          <th class="num-val">Ratio TQ / LB</th>
          <th class="num-val">Ratio WHT / LB</th>
          <th class="num-val">Ratio Outlier / LB</th>
        </tr>
      </thead>
      <tbody>
        {mse_rows}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Inner Product (IP) Distortion Comparison</h2>
    <table>
      <thead>
        <tr>
          <th>Bits (b)</th>
          <th class="num-val">Shannon IP LB</th>
          <th class="num-val">TQ IP (Flat QR)</th>
          <th class="num-val">WHT IP</th>
          <th class="num-val">Ratio TQ / LB</th>
          <th class="num-val">Ratio WHT / LB</th>
        </tr>
      </thead>
      <tbody>
        {ip_rows}
      </tbody>
    </table>
  </div>

  <div class="footer">
    Generated by distortion_bounds.py &middot; d=64 &middot; Seed: 42 &middot; Device: {DEVICE.upper()}
  </div>
</div>
</body>
</html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()
