"""
llama_qjl_sweep_and_control.py
==============================
1. Rerun the QJL vs MSE-only win-rate sweep on Llama-3.2-1B-Instruct
   at bit budgets 2, 3, 4, and 5.
   - Evaluates original (Symmetric Causal Mask) win-rate.
   - Evaluates corrected (Asymmetric Diagonal) win-rate.
2. Run synthetic positive-control case (d=1536, bits=2) under different correlations (rho=0.0 and rho=0.8)
   to confirm the scoring logic and show that QJL produces a non-zero win-rate when appropriate.

Usage:
  python scratch/llama_qjl_sweep_and_control.py [--hf_token HF_TOKEN]
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from turboquant.turbo_quant_demo import TurboQuantMSE, TurboQuantProd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Llama-1B multi-bit sweep & positive control")
parser.add_argument("--hf_token", type=str, default=None,
                    help="HuggingFace access token (for Llama model). Also reads from HF_TOKEN env var.")
args = parser.parse_args()

hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)
if hf_token:
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(project_dir, "results")
os.makedirs(os.path.join(results_dir, "llama_1b"), exist_ok=True)
os.makedirs(os.path.join(results_dir, "theory"), exist_ok=True)

# ---------------------------------------------------------------------------
# 4 diverse 128+ token prompts
# ---------------------------------------------------------------------------
PROMPTS = [
    (
        "The human brain is the most complex organ in the known universe. It contains approximately 86 billion "
        "neurons, each forming thousands of synaptic connections with neighboring cells. This intricate network "
        "allows the brain to process sensory information, regulate bodily functions, store memories, and generate "
        "conscious thought. Modern neuroscience has made remarkable strides in understanding the architecture of "
        "the brain, but many fundamental questions remain unanswered. How do distributed neural circuits give "
        "rise to unified subjective experience? What is the precise mechanism of long-term memory consolidation? "
        "How does the brain distinguish relevant signals from irrelevant noise in real time? These questions drive "
        "ongoing research across cognitive science, computational neuroscience, and artificial intelligence. "
        "The parallels between biological neural networks and deep learning architectures continue to inspire "
        "new ideas in both fields."
    ),
    (
        "Climate change represents one of the most pressing challenges of the twenty-first century. Rising "
        "global temperatures, driven by the accumulation of greenhouse gases in the atmosphere, are already "
        "causing measurable shifts in weather patterns, sea levels, and biodiversity. The Intergovernmental "
        "Panel on Climate Change has repeatedly warned that limiting warming to 1.5 degrees Celsius above "
        "pre-industrial levels requires rapid, unprecedented reductions in carbon dioxide and methane emissions. "
        "Renewable energy technologies such as solar photovoltaics and wind turbines have achieved dramatic "
        "cost reductions over the past decade, making clean electricity increasingly competitive with fossil "
        "fuels. However, decarbonizing heavy industry, agriculture, and aviation presents formidable technical "
        "and economic obstacles. Carbon capture and storage, next-generation nuclear power, and green hydrogen "
        "are among the technologies being explored to fill gaps that electrification alone cannot bridge. "
        "International cooperation remains essential, as no single nation can address this global challenge."
    ),
    (
        "The history of mathematics is a story of expanding abstraction. Ancient civilizations developed "
        "arithmetic for commerce and astronomy, while Greek mathematicians introduced the concept of proof "
        "and rigorous deduction. The invention of calculus by Newton and Leibniz in the seventeenth century "
        "provided the mathematical language for classical mechanics and much of physics. The nineteenth "
        "century saw an explosion of new structures: non-Euclidean geometry challenged the uniqueness of "
        "Euclid's axioms, abstract algebra generalized the properties of numbers to groups and rings, and "
        "set theory provided a unified foundation for all of mathematics. The twentieth century brought "
        "revolutionary results such as Gödel's incompleteness theorems, which showed that any sufficiently "
        "powerful formal system contains true statements that cannot be proved within that system. Today, "
        "mathematics underpins fields as diverse as cryptography, quantum mechanics, machine learning, "
        "and financial modeling, and new connections between seemingly unrelated branches are discovered every year."
    ),
    (
        "Large language models have transformed natural language processing over the past several years. "
        "Unlike earlier recurrent or convolutional architectures, transformer-based models rely on the "
        "self-attention mechanism to model dependencies between all pairs of tokens in a sequence simultaneously. "
        "This architectural choice enables efficient parallelization during training and allows models to "
        "capture long-range relationships that earlier systems struggled with. Scaling laws suggest that "
        "model performance improves predictably with increases in both the number of parameters and the "
        "volume of training data. Emergent capabilities such as in-context learning, chain-of-thought "
        "reasoning, and code generation have been observed at certain scale thresholds, though the underlying "
        "mechanisms remain active areas of research. Efficient inference is a major practical concern: "
        "compressing the key-value cache through quantization, pruning, or low-rank approximation reduces "
        "memory bandwidth requirements and enables deployment of large models on resource-constrained hardware. "
        "The interplay between model scale, architecture design, and quantization error is central to this work."
    ),
]

# ---------------------------------------------------------------------------
# 1. Load model and tokenizer
# ---------------------------------------------------------------------------
model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {model_id} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if device == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)
model.eval()

n_layer = model.config.num_hidden_layers
n_q_heads = model.config.num_attention_heads
n_kv_heads = getattr(model.config, "num_key_value_heads", n_q_heads)
d_head = model.config.hidden_size // n_q_heads
gqa_ratio = n_q_heads // n_kv_heads

print(f"Llama-1B architecture: {n_layer} layers, {n_q_heads} Q-heads, {n_kv_heads} KV-heads, d_head={d_head}, GQA={gqa_ratio}:1")

# ---------------------------------------------------------------------------
# 2. Extract Q and K projections
# ---------------------------------------------------------------------------
def run_prompt_llama(prompt_text):
    q_captures = {}
    k_captures = {}
    hooks = []

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        def make_q_hook(li):
            def hook_fn(module, inp, out):
                q_captures[li] = out.detach().float()
            return hook_fn

        def make_k_hook(li):
            def hook_fn(module, inp, out):
                k_captures[li] = out.detach().float()
            return hook_fn

        hooks.append(attn.q_proj.register_forward_hook(make_q_hook(layer_idx)))
        hooks.append(attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))

    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    if device == "cuda":
        input_ids = input_ids.cuda()

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    all_q_list = []
    all_k_list = []

    for li in range(n_layer):
        q_raw = q_captures[li][0].cpu()
        k_raw = k_captures[li][0].cpu()

        seq_len = q_raw.shape[0]
        q_heads = q_raw.view(seq_len, n_q_heads, d_head).permute(1, 0, 2)
        k_heads = k_raw.view(seq_len, n_kv_heads, d_head).permute(1, 0, 2)

        all_q_list.append(q_heads)
        all_k_list.append(k_heads)

    return all_q_list, all_k_list

print(f"\nRunning {len(PROMPTS)} prompts to extract real Q/K vectors...")
per_prompt_q = []
per_prompt_k = []
for idx, prompt in enumerate(PROMPTS):
    token_count = len(tokenizer.encode(prompt))
    print(f"  Prompt {idx+1}: {token_count} tokens")
    q_list, k_list = run_prompt_llama(prompt)
    per_prompt_q.append(q_list)
    per_prompt_k.append(k_list)

# ---------------------------------------------------------------------------
# 3. Rerun Sweep for bit budgets [2, 3, 4, 5]
# ---------------------------------------------------------------------------
print("\n=== Starting Win-Rate Sweep over Bit Budgets [2, 3, 4, 5] ===")
results_by_budget = {}

for bits in [2, 3, 4, 5]:
    print(f"Evaluating {bits}-bit budget...")
    mse_q = TurboQuantMSE(d_head, bits=bits, device="cpu", seed=SEED)
    qjl_q = TurboQuantProd(d_head, bits=bits, device="cpu", seed=SEED)

    # Accumulators for Symmetric Causal Mask
    sym_mse_acc = np.zeros((n_layer, n_kv_heads, len(PROMPTS)))
    sym_qjl_acc = np.zeros((n_layer, n_kv_heads, len(PROMPTS)))

    # Accumulators for Asymmetric Diagonal
    asym_mse_acc = np.zeros((n_layer, n_kv_heads, len(PROMPTS)))
    asym_qjl_acc = np.zeros((n_layer, n_kv_heads, len(PROMPTS)))

    for p_idx in range(len(PROMPTS)):
        for l in range(n_layer):
            k_all = per_prompt_k[p_idx][l].float()
            q_all = per_prompt_q[p_idx][l].float()
            seq_len = k_all.shape[1]
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

            for kv_h in range(n_kv_heads):
                k = k_all[kv_h]
                q_group_start = kv_h * gqa_ratio
                q = q_all[q_group_start]

                # Quantize K
                ind_mse_k, norm_mse_k = mse_q.quantize(k)
                k_hat_mse = mse_q.dequantize(ind_mse_k, norm_mse_k).float()
                ind_qjl_k, norm_qjl_k, signs_qjl_k, gamma_qjl_k = qjl_q.quantize(k)
                k_hat_qjl = qjl_q.dequantize(ind_qjl_k, norm_qjl_k, signs_qjl_k, gamma_qjl_k).float()

                # Quantize Q (Symmetric only)
                ind_mse_q, norm_mse_q = mse_q.quantize(q)
                q_hat_mse = mse_q.dequantize(ind_mse_q, norm_mse_q).float()
                ind_qjl_q, norm_qjl_q, signs_qjl_q, gamma_qjl_q = qjl_q.quantize(q)
                q_hat_qjl = qjl_q.dequantize(ind_qjl_q, norm_qjl_q, signs_qjl_q, gamma_qjl_q).float()

                # 1. Symmetric Causal Mask Scoring
                true_ips = (q @ k.T)[mask]
                ips_mse_sym = (q_hat_mse @ k_hat_mse.T)[mask]
                ips_qjl_sym = (q_hat_qjl @ k_hat_qjl.T)[mask]
                sym_mse_acc[l, kv_h, p_idx] = torch.mean((ips_mse_sym - true_ips)**2).item()
                sym_qjl_acc[l, kv_h, p_idx] = torch.mean((ips_qjl_sym - true_ips)**2).item()

                # 2. Asymmetric Diagonal Scoring (matching tokens)
                true_ips_diag = torch.diag(q @ k.T)
                ips_mse_asym = torch.diag(q @ k_hat_mse.T)
                ips_qjl_asym = torch.diag(q @ k_hat_qjl.T)
                asym_mse_acc[l, kv_h, p_idx] = torch.mean((ips_mse_asym - true_ips_diag)**2).item()
                asym_qjl_acc[l, kv_h, p_idx] = torch.mean((ips_qjl_asym - true_ips_diag)**2).item()

    # Average over prompts and compute win rates
    sym_mse_mean = sym_mse_acc.mean(axis=2)
    sym_qjl_mean = sym_qjl_acc.mean(axis=2)
    sym_qjl_better = sym_qjl_mean < sym_mse_mean
    sym_wins = int(sym_qjl_better.sum())

    asym_mse_mean = asym_mse_acc.mean(axis=2)
    asym_qjl_mean = asym_qjl_acc.mean(axis=2)
    asym_qjl_better = asym_qjl_mean < asym_mse_mean
    asym_wins = int(asym_qjl_better.sum())

    total_cells = n_layer * n_kv_heads
    print(f"  Symmetric Causal Mask wins: {sym_wins} / {total_cells} ({sym_wins/total_cells*100:.2f}%)")
    print(f"  Asymmetric Diagonal wins:    {asym_wins} / {total_cells} ({asym_wins/total_cells*100:.2f}%)")

    results_by_budget[bits] = {
        "symmetric": {
            "wins": sym_wins,
            "total_cells": total_cells,
            "win_rate": sym_wins / total_cells,
            "mse_mean_all": float(sym_mse_mean.mean()),
            "qjl_mean_all": float(sym_qjl_mean.mean()),
        },
        "asymmetric_diagonal": {
            "wins": asym_wins,
            "total_cells": total_cells,
            "win_rate": asym_wins / total_cells,
            "mse_mean_all": float(asym_mse_mean.mean()),
            "qjl_mean_all": float(asym_qjl_mean.mean()),
        }
    }

# Save Llama results
out_json_path = os.path.join(results_dir, "llama_1b", "multibit_winrate.json")
with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump(results_by_budget, f, indent=2)
print(f"Saved Llama multibit results to {out_json_path}")

# ---------------------------------------------------------------------------
# 4. Synthetic Positive-Control Case (d=1536, bits=2)
# ---------------------------------------------------------------------------
print("\n=== Running Synthetic Positive-Control sweeps (d=1536, bits=2) ===")

d_syn = 1536
bits_syn = 2
n_layer_syn = 16
n_heads_syn = 8
n_prompts_syn = 4
seq_len_syn = 128

mse_q_syn = TurboQuantMSE(d_syn, bits=bits_syn, device="cpu", seed=SEED)
qjl_q_syn = TurboQuantProd(d_syn, bits=bits_syn, device="cpu", seed=SEED)

synthetic_results = {}

for rho in [0.0, 0.8]:
    print(f"Running synthetic control with correlation rho = {rho}...")
    sym_mse_acc = np.zeros((n_layer_syn, n_heads_syn, n_prompts_syn))
    sym_qjl_acc = np.zeros((n_layer_syn, n_heads_syn, n_prompts_syn))
    asym_mse_acc = np.zeros((n_layer_syn, n_heads_syn, n_prompts_syn))
    asym_qjl_acc = np.zeros((n_layer_syn, n_heads_syn, n_prompts_syn))

    for p_idx in range(n_prompts_syn):
        for l in range(n_layer_syn):
            for h in range(n_heads_syn):
                q_raw = torch.randn(seq_len_syn, d_syn)
                eps = torch.randn(seq_len_syn, d_syn)
                k_raw = rho * q_raw + np.sqrt(1 - rho**2) * eps

                q = q_raw / (torch.norm(q_raw, dim=-1, keepdim=True) + 1e-8)
                k = k_raw / (torch.norm(k_raw, dim=-1, keepdim=True) + 1e-8)

                # Quantize K
                ind_mse_k, norm_mse_k = mse_q_syn.quantize(k)
                k_hat_mse = mse_q_syn.dequantize(ind_mse_k, norm_mse_k).float()
                ind_qjl_k, norm_qjl_k, signs_qjl_k, gamma_qjl_k = qjl_q_syn.quantize(k)
                k_hat_qjl = qjl_q_syn.dequantize(ind_qjl_k, norm_qjl_k, signs_qjl_k, gamma_qjl_k).float()

                # Quantize Q (Symmetric only)
                ind_mse_q, norm_mse_q = mse_q_syn.quantize(q)
                q_hat_mse = mse_q_syn.dequantize(ind_mse_q, norm_mse_q).float()
                ind_qjl_q, norm_qjl_q, signs_qjl_q, gamma_qjl_q = qjl_q_syn.quantize(q)
                q_hat_qjl = qjl_q_syn.dequantize(ind_qjl_q, norm_qjl_q, signs_qjl_q, gamma_qjl_q).float()

                # 1. Symmetric Causal Mask
                mask = torch.tril(torch.ones(seq_len_syn, seq_len_syn, dtype=torch.bool))
                true_ips = (q @ k.T)[mask]
                ips_mse_sym = (q_hat_mse @ k_hat_mse.T)[mask]
                ips_qjl_sym = (q_hat_qjl @ k_hat_qjl.T)[mask]
                sym_mse_acc[l, h, p_idx] = torch.mean((ips_mse_sym - true_ips)**2).item()
                sym_qjl_acc[l, h, p_idx] = torch.mean((ips_qjl_sym - true_ips)**2).item()

                # 2. Asymmetric Diagonal
                true_ips_diag = torch.diag(q @ k.T)
                ips_mse_asym = torch.diag(q @ k_hat_mse.T)
                ips_qjl_asym = torch.diag(q @ k_hat_qjl.T)
                asym_mse_acc[l, h, p_idx] = torch.mean((ips_mse_asym - true_ips_diag)**2).item()
                asym_qjl_acc[l, h, p_idx] = torch.mean((ips_qjl_asym - true_ips_diag)**2).item()

    sym_mse_mean = sym_mse_acc.mean(axis=2)
    sym_qjl_mean = sym_qjl_acc.mean(axis=2)
    sym_qjl_better = sym_qjl_mean < sym_mse_mean
    sym_wins = int(sym_qjl_better.sum())

    asym_mse_mean = asym_mse_acc.mean(axis=2)
    asym_qjl_mean = asym_qjl_acc.mean(axis=2)
    asym_qjl_better = asym_qjl_mean < asym_mse_mean
    asym_wins = int(asym_qjl_better.sum())

    total_cells_syn = n_layer_syn * n_heads_syn
    print(f"  Rho = {rho} | Symmetric Causal Mask wins: {sym_wins} / {total_cells_syn} ({sym_wins/total_cells_syn*100:.2f}%)")
    print(f"  Rho = {rho} | Asymmetric Diagonal wins:    {asym_wins} / {total_cells_syn} ({asym_wins/total_cells_syn*100:.2f}%)")

    synthetic_results[str(rho)] = {
        "symmetric": {
            "wins": sym_wins,
            "total_cells": total_cells_syn,
            "win_rate": sym_wins / total_cells_syn,
            "mse_mean_all": float(sym_mse_mean.mean()),
            "qjl_mean_all": float(sym_qjl_mean.mean()),
        },
        "asymmetric_diagonal": {
            "wins": asym_wins,
            "total_cells": total_cells_syn,
            "win_rate": asym_wins / total_cells_syn,
            "mse_mean_all": float(asym_mse_mean.mean()),
            "qjl_mean_all": float(asym_qjl_mean.mean()),
        }
    }

syn_json_path = os.path.join(results_dir, "theory", "positive_control.json")
with open(syn_json_path, "w", encoding="utf-8") as f:
    json.dump(synthetic_results, f, indent=2)
print(f"Saved positive-control results to {syn_json_path}")

# ---------------------------------------------------------------------------
# 5. Output Summary Table
# ---------------------------------------------------------------------------
print("\n" + "="*80)
print(f"{'EVALUATION CASE':<45} | {'SYM CAUSAL WR':<15} | {'ASYM DIAG WR':<15}")
print("="*80)
for bits in [2, 3, 4, 5]:
    sym_wr = results_by_budget[bits]["symmetric"]["win_rate"]
    asym_wr = results_by_budget[bits]["asymmetric_diagonal"]["win_rate"]
    print(f"Llama-3.2-1B ({bits}-bit)                             | {sym_wr*100:>12.2f}% | {asym_wr*100:>12.2f}%")
print("-"*80)
sym_wr_syn_0 = synthetic_results["0.0"]["symmetric"]["win_rate"]
asym_wr_syn_0 = synthetic_results["0.0"]["asymmetric_diagonal"]["win_rate"]
print(f"Synthetic Control (d=1536, 2b, independent rho=0.0)   | {sym_wr_syn_0*100:>12.2f}% | {asym_wr_syn_0*100:>12.2f}%")

sym_wr_syn_8 = synthetic_results["0.8"]["symmetric"]["win_rate"]
asym_wr_syn_8 = synthetic_results["0.8"]["asymmetric_diagonal"]["win_rate"]
print(f"Synthetic Control (d=1536, 2b, correlated rho=0.8)    | {sym_wr_syn_8*100:>12.2f}% | {asym_wr_syn_8*100:>12.2f}%")
print("="*80)

print("\nScript completed successfully!")
