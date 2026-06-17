"""
nn_search_eval_llama.py
=======================
NN recall benchmark at Llama-3.2 head dimensions.

  Llama-3.2-1B-Instruct: d_head = 64   -> results/llama_1b/nn_search.json
  Llama-3.2-3B-Instruct: d_head = 128  -> results/llama_3b/nn_search.json

This benchmark is purely synthetic (no model loading needed) — it evaluates
quantization quality at a specific dimension. Results use the same JSON schema
as the original nn_search_eval.py so they compare directly.

Usage:
  python benchmarks/nn_search_eval_llama.py --dim 64
  python benchmarks/nn_search_eval_llama.py --dim 128
  python benchmarks/nn_search_eval_llama.py --dim both   (runs 64 and 128)
"""

import argparse
import os
import sys
import json
import time
import torch
import numpy as np
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from turboquant.turbo_quant_demo import TurboQuantProd as OrigTurboQuantProd
from turboquant.wht_quantizer import AdaptiveTurboQuantProd
from turboquant.wht_quantizer import TurboQuantMSE as WhtTurboQuantMSE
from turboquant.outlier_channel_quantizer import OutlierChannelQuantizer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="NN recall at Llama KV-head dimensions")
parser.add_argument("--dim", type=str, default="both",
                    choices=["64", "128", "both"],
                    help="Head dimension to evaluate (64=1B, 128=3B, both=run both)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Seed / device
# ---------------------------------------------------------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Wrappers  (copied from nn_search_eval.py, no changes needed)
# ============================================================
class FlatTurboQuantWrapper:
    def __init__(self, dim, bits, device, seed=0):
        self.dim, self.bits, self.device = dim, bits, device
        self.quantizer = OrigTurboQuantProd(dim, bits, device, seed=seed)

    def quantize(self, x):
        return self.quantizer.quantize(x)

    def dequantize(self, compressed):
        return self.quantizer.dequantize(*compressed)

    def bits_per_vector(self):
        return self.dim * (self.bits - 1) + self.dim * 1 + 32 + 32


class WhtAsymWrapper:
    def __init__(self, dim, bits, device, seed=0):
        self.dim, self.bits, self.device = dim, bits, device
        self.quantizer = AdaptiveTurboQuantProd(dim, bits, device, seed=seed)

    def quantize(self, x):
        return self.quantizer.quantize(x)

    def dequantize(self, compressed):
        return self.quantizer.dequantize(*compressed)

    def bits_per_vector(self):
        if self.quantizer.use_qjl:
            return self.dim * (self.bits - 1) + self.dim * 1 + 32 + 32
        else:
            return self.dim * self.bits + 32


class OutlierSplittingWrapper:
    def __init__(self, dim, avg_bits, device, seed=0):
        self.dim, self.avg_bits, self.device = dim, avg_bits, device
        self.quantizer = OutlierChannelQuantizer(dim, avg_bits, device, seed=seed)

    def calibrate(self, cal_vectors):
        self.quantizer.calibrate(cal_vectors)

    def quantize(self, x):
        return self.quantizer.quantize(x)

    def dequantize(self, compressed):
        return self.quantizer.dequantize(compressed)

    def bits_per_vector(self):
        o = self.quantizer.outlier_count
        n = self.dim - o
        return o * self.quantizer.outlier_bits + n * self.quantizer.normal_bits + 32


# ============================================================
# Evaluation function  (identical to nn_search_eval.py)
# ============================================================
def evaluate_scheme(db, queries, true_top1, true_top10, wrapper) -> Dict:
    n_queries = queries.shape[0]
    n_db = db.shape[0]
    chunk_size = 1000

    compressed_db = []
    for i in range(0, n_db, chunk_size):
        compressed_db.append(wrapper.quantize(db[i:i+chunk_size]))

    recon_db = []
    for chunk in compressed_db:
        recon_db.append(wrapper.dequantize(chunk).float())
    recon_db = torch.cat(recon_db, dim=0)

    recon_queries = []
    for i in range(0, n_queries, chunk_size):
        q_chunk = wrapper.quantize(queries[i:i+chunk_size])
        recon_queries.append(wrapper.dequantize(q_chunk).float())
    recon_queries = torch.cat(recon_queries, dim=0)

    approx_ip = torch.matmul(recon_queries, recon_db.T)
    approx_ranks = torch.argsort(approx_ip, dim=-1, descending=True)
    approx_top1 = approx_ranks[:, 0]

    recall_1 = recall_10 = 0
    ranks = []
    for q_idx in range(n_queries):
        target = true_top1[q_idx].item()
        if approx_top1[q_idx].item() == target:
            recall_1 += 1
        if target in approx_ranks[q_idx, :10]:
            recall_10 += 1
        true_rank = (approx_ranks[q_idx] == target).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(true_rank)

    recall_1  = (recall_1  / n_queries) * 100.0
    recall_10 = (recall_10 / n_queries) * 100.0
    mean_rank = float(np.mean(ranks))

    db_bits_fp16       = n_db * db.shape[1] * 16
    db_bits_compressed = n_db * wrapper.bits_per_vector()
    comp_ratio = db_bits_fp16 / db_bits_compressed

    return {
        "recall_1": recall_1,
        "recall_10": recall_10,
        "mean_rank": mean_rank,
        "compression_ratio": comp_ratio,
    }


# ============================================================
# Per-dimension runner
# ============================================================
DIM_TO_MODEL = {64: "Llama-3.2-1B-Instruct", 128: "Llama-3.2-3B-Instruct"}
DIM_TO_DIR   = {64: "llama_1b",               128: "llama_3b"}

def run_for_dim(dim: int):
    print(f"\n{'='*70}")
    print(f"  NN Search Benchmark at d_head={dim}  ({DIM_TO_MODEL[dim]})")
    print(f"{'='*70}")

    n_db    = 10_000
    n_query = 500

    torch.manual_seed(SEED)
    G_db = torch.randn(n_db, dim, dtype=torch.float32, device=DEVICE)
    db   = (G_db / (torch.norm(G_db, dim=-1, keepdim=True) + 1e-8)).to(DTYPE)

    torch.manual_seed(1)
    G_q     = torch.randn(n_query, dim, dtype=torch.float32, device=DEVICE)
    queries = (G_q / (torch.norm(G_q, dim=-1, keepdim=True) + 1e-8)).to(DTYPE)

    exact_ip   = torch.matmul(queries.float(), db.float().T)
    true_ranks = torch.argsort(exact_ip, dim=-1, descending=True)
    true_top1  = true_ranks[:, 0]
    true_top10 = true_ranks[:, :10]

    results = {}

    # --- Flat TurboQuant ---
    print("\nScheme 1: Flat TurboQuant (Symmetric QJL)")
    results["flat_turboquant"] = {}
    for b in [2, 3, 4]:
        w = FlatTurboQuantWrapper(dim, b, DEVICE, seed=SEED)
        res = evaluate_scheme(db, queries, true_top1, true_top10, w)
        results["flat_turboquant"][str(b)] = res
        print(f"  {b}-bit -> R@1={res['recall_1']:.2f}%  R@10={res['recall_10']:.2f}%  "
              f"MeanRank={res['mean_rank']:.1f}  CR={res['compression_ratio']:.2f}x")

    # --- WHT + Asym ---
    print("\nScheme 2: WHT + Asymmetric QJL")
    results["wht_asym"] = {}
    for b in [2, 3, 4]:
        w = WhtAsymWrapper(dim, b, DEVICE, seed=SEED)
        res = evaluate_scheme(db, queries, true_top1, true_top10, w)
        results["wht_asym"][str(b)] = res
        print(f"  {b}-bit -> R@1={res['recall_1']:.2f}%  R@10={res['recall_10']:.2f}%  "
              f"MeanRank={res['mean_rank']:.1f}  CR={res['compression_ratio']:.2f}x")

    # --- Outlier Splitting ---
    print("\nScheme 3: Outlier Channel Splitting")
    results["outlier_split"] = {}
    for b_avg in [2.5, 3.5]:
        w = OutlierSplittingWrapper(dim, b_avg, DEVICE, seed=SEED)
        w.calibrate(db[:1000])
        res = evaluate_scheme(db, queries, true_top1, true_top10, w)
        results["outlier_split"][str(b_avg)] = res
        print(f"  {b_avg}-bit -> R@1={res['recall_1']:.2f}%  R@10={res['recall_10']:.2f}%  "
              f"MeanRank={res['mean_rank']:.1f}  CR={res['compression_ratio']:.2f}x")

    # Save JSON
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_dir, "results", DIM_TO_DIR[dim])
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "nn_search.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    dims_to_run = []
    if args.dim == "both":
        dims_to_run = [64, 128]
    else:
        dims_to_run = [int(args.dim)]

    for d in dims_to_run:
        run_for_dim(d)

    print("\nAll NN recall benchmarks completed.")
