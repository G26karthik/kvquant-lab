import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import math
import time
import torch
import numpy as np
from typing import Dict, List, Tuple

# Import quantizers
from turboquant.turbo_quant_demo import TurboQuantProd as OrigTurboQuantProd
from turboquant.wht_quantizer import AdaptiveTurboQuantProd
from turboquant.wht_quantizer import TurboQuantMSE as WhtTurboQuantMSE
from turboquant.outlier_channel_quantizer import OutlierChannelQuantizer

# Constants
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ----------------------------------------------------------------------
# Flat TurboQuant (QR + symmetric QJL, original) wrapper
# ----------------------------------------------------------------------
class FlatTurboQuantWrapper:
    def __init__(self, dim: int, bits: int, device: str, seed: int = 0):
        self.dim = dim
        self.bits = bits
        self.device = device
        # Original QR-based Prod quantizer
        self.quantizer = OrigTurboQuantProd(dim, bits, device, seed=seed)

    def quantize(self, x: torch.Tensor):
        return self.quantizer.quantize(x)

    def dequantize(self, compressed):
        return self.quantizer.dequantize(*compressed)

    def bits_per_vector(self) -> float:
        # indices: bits-1, QJL signs: 1, norms: 32, gamma: 32
        return self.dim * (self.bits - 1) + self.dim * 1 + 32 + 32


# ----------------------------------------------------------------------
# WHT + Asymmetric QJL wrapper
# ----------------------------------------------------------------------
class WhtAsymWrapper:
    def __init__(self, dim: int, bits: int, device: str, seed: int = 0):
        self.dim = dim
        self.bits = bits
        self.device = device
        # Uses AdaptiveTurboQuantProd (which falls back to MSE-only for d=64)
        self.quantizer = AdaptiveTurboQuantProd(dim, bits, device, seed=seed)

    def quantize(self, x: torch.Tensor):
        return self.quantizer.quantize(x)

    def dequantize(self, compressed):
        return self.quantizer.dequantize(*compressed)

    def bits_per_vector(self) -> float:
        # If QJL is active (dim >= 128)
        if self.quantizer.use_qjl:
            return self.dim * (self.bits - 1) + self.dim * 1 + 32 + 32
        else:
            # MSE-only at d=64
            return self.dim * self.bits + 32


# ----------------------------------------------------------------------
# Outlier Channel Splitting wrapper
# ----------------------------------------------------------------------
class OutlierSplittingWrapper:
    def __init__(self, dim: int, avg_bits: float, device: str, seed: int = 0):
        self.dim = dim
        self.avg_bits = avg_bits
        self.device = device
        self.quantizer = OutlierChannelQuantizer(dim, avg_bits, device, seed=seed)

    def calibrate(self, cal_vectors: torch.Tensor):
        self.quantizer.calibrate(cal_vectors)

    def quantize(self, x: torch.Tensor):
        return self.quantizer.quantize(x)

    def dequantize(self, compressed):
        return self.quantizer.dequantize(compressed)

    def bits_per_vector(self) -> float:
        # outlier_count * outlier_bits + normal_count * normal_bits + 32
        o_count = self.quantizer.outlier_count
        n_count = self.dim - o_count
        return o_count * self.quantizer.outlier_bits + n_count * self.quantizer.normal_bits + 32


# ----------------------------------------------------------------------
# Evaluation Function
# ----------------------------------------------------------------------
def evaluate_scheme(db: torch.Tensor, queries: torch.Tensor,
                    true_top1: torch.Tensor, true_top10: torch.Tensor,
                    wrapper) -> Dict:
    n_queries = queries.shape[0]
    n_db = db.shape[0]
    
    # Compress database
    compressed_db = []
    # Process in chunks to prevent GPU out-of-memory if needed
    chunk_size = 1000
    for i in range(0, n_db, chunk_size):
        chunk = db[i:i+chunk_size]
        q_chunk = wrapper.quantize(chunk)
        compressed_db.append(q_chunk)
        
    # Reconstruct database
    recon_db = []
    for chunk in compressed_db:
        r_chunk = wrapper.dequantize(chunk)
        recon_db.append(r_chunk.float())
    recon_db = torch.cat(recon_db, dim=0) # (10000, 64)
    
    # Compress and reconstruct queries
    recon_queries = []
    for i in range(0, n_queries, chunk_size):
        chunk = queries[i:i+chunk_size]
        q_chunk = wrapper.quantize(chunk)
        r_chunk = wrapper.dequantize(q_chunk)
        recon_queries.append(r_chunk.float())
    recon_queries = torch.cat(recon_queries, dim=0) # (500, 64)
    
    # Search using approximate inner products
    # Shape: (500, 10000)
    approx_ip = torch.matmul(recon_queries, recon_db.T)
    
    # Find approx nearest neighbors
    approx_ranks = torch.argsort(approx_ip, dim=-1, descending=True)
    approx_top1 = approx_ranks[:, 0]
    
    # Calculate metrics
    recall_1 = 0
    recall_10 = 0
    ranks = []
    
    for q_idx in range(n_queries):
        target_top1 = true_top1[q_idx].item()
        
        # Recall@1
        if approx_top1[q_idx].item() == target_top1:
            recall_1 += 1
            
        # Recall@10
        if target_top1 in approx_ranks[q_idx, :10]:
            recall_10 += 1
            
        # Rank of the true top-1 in approx ranking
        true_rank = (approx_ranks[q_idx] == target_top1).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(true_rank)
        
    recall_1 = (recall_1 / n_queries) * 100.0
    recall_10 = (recall_10 / n_queries) * 100.0
    mean_rank = float(np.mean(ranks))
    
    # Compression Ratio
    db_bits_fp16 = n_db * db.shape[1] * 16
    db_bits_compressed = n_db * wrapper.bits_per_vector()
    comp_ratio = db_bits_fp16 / db_bits_compressed
    
    return {
        "recall_1": recall_1,
        "recall_10": recall_10,
        "mean_rank": mean_rank,
        "compression_ratio": comp_ratio
    }


# ----------------------------------------------------------------------
# Main Orchestrator
# ----------------------------------------------------------------------
def main():
    print("=" * 75)
    print("  Nearest Neighbor Search Evaluation Suite (d=64)")
    print("=" * 75)
    
    dim = 64
    n_db = 10000
    n_query = 500
    
    # 1. Generate Database (seed=42) and Queries (seed=1)
    # Database
    torch.manual_seed(SEED)
    G_db = torch.randn(n_db, dim, dtype=torch.float32, device=DEVICE)
    db = G_db / (torch.norm(G_db, dim=-1, keepdim=True) + 1e-8)
    db = db.to(DTYPE)
    
    # Queries
    torch.manual_seed(1)
    G_q = torch.randn(n_query, dim, dtype=torch.float32, device=DEVICE)
    queries = G_q / (torch.norm(G_q, dim=-1, keepdim=True) + 1e-8)
    queries = queries.to(DTYPE)
    
    # 2. Compute Exact Nearest Neighbors (FP16)
    print("Computing exact nearest neighbors in FP16...")
    exact_ip = torch.matmul(queries.float(), db.float().T) # (500, 10000)
    true_ranks = torch.argsort(exact_ip, dim=-1, descending=True)
    true_top1 = true_ranks[:, 0]
    true_top10 = true_ranks[:, :10]
    
    results = {}
    
    # --- Scheme 1: Flat TurboQuant (Symmetric QJL) ---
    print("\nEvaluating Scheme 1: Flat TurboQuant (Symmetric QJL)...")
    results["flat_turboquant"] = {}
    for b in [2, 3, 4]:
        wrapper = FlatTurboQuantWrapper(dim, b, DEVICE, seed=SEED)
        res = evaluate_scheme(db, queries, true_top1, true_top10, wrapper)
        results["flat_turboquant"][str(b)] = res
        print(f"  {b}-bit -> Recall@1: {res['recall_1']:.2f}%, Recall@10: {res['recall_10']:.2f}%, Mean Rank: {res['mean_rank']:.2f}, CR: {res['compression_ratio']:.2f}x")
        
    # --- Scheme 2: WHT + Asymmetric QJL ---
    print("\nEvaluating Scheme 2: WHT + Asymmetric QJL...")
    results["wht_asym"] = {}
    for b in [2, 3, 4]:
        wrapper = WhtAsymWrapper(dim, b, DEVICE, seed=SEED)
        res = evaluate_scheme(db, queries, true_top1, true_top10, wrapper)
        results["wht_asym"][str(b)] = res
        print(f"  {b}-bit -> Recall@1: {res['recall_1']:.2f}%, Recall@10: {res['recall_10']:.2f}%, Mean Rank: {res['mean_rank']:.2f}, CR: {res['compression_ratio']:.2f}x")
        
    # --- Scheme 3: Outlier Channel Splitting ---
    print("\nEvaluating Scheme 3: Outlier Channel Splitting...")
    results["outlier_split"] = {}
    for b_avg in [2.5, 3.5]:
        wrapper = OutlierSplittingWrapper(dim, b_avg, DEVICE, seed=SEED)
        # Calibrate using database vectors
        wrapper.calibrate(db[:1000])
        res = evaluate_scheme(db, queries, true_top1, true_top10, wrapper)
        results["outlier_split"][str(b_avg)] = res
        print(f"  {b_avg}-bit -> Recall@1: {res['recall_1']:.2f}%, Recall@10: {res['recall_10']:.2f}%, Mean Rank: {res['mean_rank']:.2f}, CR: {res['compression_ratio']:.2f}x")
        
    # 3. Save to JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    json_path = os.path.join(results_dir, "nn_search.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to {json_path}")
    
    # Generate HTML
    html_path = os.path.join(results_dir, "nn_search.html")
    generate_html(results, html_path)
    print(f"HTML report saved to {html_path}")


def generate_html(results: Dict, output_path: str):
    rows = ""
    
    # Flat TurboQuant
    for b in ["2", "3", "4"]:
        res = results["flat_turboquant"][b]
        rows += f"""
        <tr>
            <td class="dim-val">Flat TurboQuant ({b}-bit)</td>
            <td class="num-val">{res['recall_1']:.2f}%</td>
            <td class="num-val">{res['recall_10']:.2f}%</td>
            <td class="num-val">{res['mean_rank']:.2f}</td>
            <td class="num-val highlight">{res['compression_ratio']:.2f}x</td>
        </tr>
        """
        
    # WHT + Asymmetric
    for b in ["2", "3", "4"]:
        res = results["wht_asym"][b]
        rows += f"""
        <tr>
            <td class="dim-val">WHT + Asymmetric ({b}-bit)</td>
            <td class="num-val">{res['recall_1']:.2f}%</td>
            <td class="num-val">{res['recall_10']:.2f}%</td>
            <td class="num-val">{res['mean_rank']:.2f}</td>
            <td class="num-val highlight">{res['compression_ratio']:.2f}x</td>
        </tr>
        """
        
    # Outlier Splitting
    for b in ["2.5", "3.5"]:
        res = results["outlier_split"][b]
        rows += f"""
        <tr>
            <td class="dim-val">Outlier Splitting ({b}-bit)</td>
            <td class="num-val">{res['recall_1']:.2f}%</td>
            <td class="num-val">{res['recall_10']:.2f}%</td>
            <td class="num-val">{res['mean_rank']:.2f}</td>
            <td class="num-val highlight">{res['compression_ratio']:.2f}x</td>
        </tr>
        """
        
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nearest Neighbor Search Evaluation Dashboard</title>
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
    <h1>Nearest Neighbor Search Benchmark</h1>
    <div class="subtitle">Faithful Reproduction of Table 2 from the TurboQuant Paper (d=64, N=10000 Database, 500 Queries)</div>
  </header>

  <div class="card">
    <h2>Nearest Neighbor Search Metrics</h2>
    <table>
      <thead>
        <tr>
          <th>Quantization Configuration</th>
          <th class="num-val">Recall @ 1</th>
          <th class="num-val">Recall @ 10</th>
          <th class="num-val">Mean Rank</th>
          <th class="num-val">Compression Ratio</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>

  <div class="footer">
    Generated by nn_search_eval.py &middot; GPT-2 Medium &middot; Device: {DEVICE.upper()}
  </div>
</div>
</body>
</html>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
