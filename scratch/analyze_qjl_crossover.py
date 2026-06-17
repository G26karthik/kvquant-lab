"""
analyze_qjl_crossover.py
========================
Phase 1 (updated): Evaluate TurboQuantMSE vs TurboQuantProd (QJL) directly on
real GPT-2 attention vectors.

Changes from original:
  - 4 diverse prompts of 128+ tokens (averaged results)
  - Q and K L2 norms printed per layer for scale verification
  - Per-layer, per-head breakdown retained as-is
  - Results averaged + std-dev reported across prompts
  - Optional --model CLI argument (defaults to gpt2-medium)

Usage:
  python scratch/analyze_qjl_crossover.py
  python scratch/analyze_qjl_crossover.py --model gpt2-medium
"""

import argparse
import os
import sys
import json
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to sys.path to resolve turboquant module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant.turbo_quant_demo import TurboQuantMSE, TurboQuantProd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="QJL crossover analysis on real attention vectors")
parser.add_argument("--model", type=str, default="gpt2-medium",
                    help="HuggingFace model name (default: gpt2-medium)")
args = parser.parse_args()

MODEL_NAME = args.model

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Output directory (constructed dynamically)
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(project_dir, "results", "theory")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 4 diverse prompts of 128+ tokens each
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
# 1. Load model + tokenizer
# ---------------------------------------------------------------------------
print(f"\nLoading {MODEL_NAME} on CPU to extract real Query and Key vectors...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()

# Detect model architecture
model_type = model.config.model_type  # e.g. "gpt2"
n_embd = model.config.hidden_size
n_head = model.config.num_attention_heads
d_head = n_embd // n_head
n_layer = model.config.num_hidden_layers

print(f"  Model type : {model_type}")
print(f"  Layers     : {n_layer}")
print(f"  Heads      : {n_head}")
print(f"  d_head     : {d_head}")

# ---------------------------------------------------------------------------
# 2. Hook setup — GPT-2 uses c_attn (fused QKV projection)
# ---------------------------------------------------------------------------
def run_prompt_gpt2(prompt_text):
    """Extract Q, K tensors for every layer on a GPT-2-family model."""
    c_attn_outputs = []

    def hook_fn(module, input, output):
        c_attn_outputs.append(output.detach())

    hooks = []
    for block in model.transformer.h:
        hooks.append(block.attn.c_attn.register_forward_hook(hook_fn))

    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    all_q, all_k = [], []
    for out in c_attn_outputs:
        q, k, v = out.split(n_embd, dim=-1)
        q = q.view(1, -1, n_head, d_head).transpose(1, 2)  # (1, n_head, seq, d_head)
        k = k.view(1, -1, n_head, d_head).transpose(1, 2)
        all_q.append(q[0])   # (n_head, seq, d_head)
        all_k.append(k[0])

    all_q = torch.stack(all_q, dim=0)   # (n_layer, n_head, seq, d_head)
    all_k = torch.stack(all_k, dim=0)
    return all_q, all_k


# ---------------------------------------------------------------------------
# 3. Multi-prompt extraction
# ---------------------------------------------------------------------------
print(f"\nRunning {len(PROMPTS)} prompts through {MODEL_NAME}...")

per_prompt_q = []
per_prompt_k = []

for idx, prompt in enumerate(PROMPTS):
    token_count = len(tokenizer.encode(prompt))
    print(f"  Prompt {idx+1}: {token_count} tokens")

    if model_type == "gpt2":
        q, k = run_prompt_gpt2(prompt)
    else:
        raise NotImplementedError(
            f"Model type '{model_type}' not supported in this script. "
            "Use scratch/llama_qjl_winrate.py for Llama models."
        )

    per_prompt_q.append(q)
    per_prompt_k.append(k)

# ---------------------------------------------------------------------------
# 4. Print Q and K L2 norms per layer (averaged across all prompts)
# ---------------------------------------------------------------------------
print("\n--- Q and K L2 Norms per Layer (avg over tokens × heads × prompts) ---")
print(f"{'Layer':>5}  {'||Q|| mean':>12}  {'||Q|| std':>10}  {'||K|| mean':>12}  {'||K|| std':>10}")
for l in range(n_layer):
    q_norms = []
    k_norms = []
    for p in range(len(PROMPTS)):
        q_p = per_prompt_q[p][l].float()   # (n_head, seq, d_head)
        k_p = per_prompt_k[p][l].float()
        q_norms.append(torch.norm(q_p, dim=-1).reshape(-1))  # all token-head pairs
        k_norms.append(torch.norm(k_p, dim=-1).reshape(-1))
    q_norms_all = torch.cat(q_norms)
    k_norms_all = torch.cat(k_norms)
    print(
        f"  {l:>3}  {q_norms_all.mean().item():>12.4f}  {q_norms_all.std().item():>10.4f}"
        f"  {k_norms_all.mean().item():>12.4f}  {k_norms_all.std().item():>10.4f}"
    )

# ---------------------------------------------------------------------------
# 5. Pre-instantiate quantizers (same d_head for all prompts)
# ---------------------------------------------------------------------------
mse3_q = TurboQuantMSE(d_head, bits=3, device="cpu", seed=SEED)
mse4_q = TurboQuantMSE(d_head, bits=4, device="cpu", seed=SEED)
qjl4_q = TurboQuantProd(d_head, bits=4, device="cpu", seed=SEED)

# ---------------------------------------------------------------------------
# 6. Per-layer, per-head evaluation — averaged across prompts
# ---------------------------------------------------------------------------
print(f"\nEvaluating TurboQuantMSE & QJL directly on real attention vectors...")
print(f"  Averaging across {len(PROMPTS)} prompts...")

# Accumulators: shape (n_layer, n_head, n_prompts)
mse3_acc   = np.zeros((n_layer, n_head, len(PROMPTS)))
mse4_acc   = np.zeros((n_layer, n_head, len(PROMPTS)))
qjl4_acc   = np.zeros((n_layer, n_head, len(PROMPTS)))
bias4_acc  = np.zeros((n_layer, n_head, len(PROMPTS)))
var4_acc   = np.zeros((n_layer, n_head, len(PROMPTS)))
biasq_acc  = np.zeros((n_layer, n_head, len(PROMPTS)))
varq_acc   = np.zeros((n_layer, n_head, len(PROMPTS)))

for p_idx, (all_q, all_k) in enumerate(zip(per_prompt_q, per_prompt_k)):
    seq_len = all_q.shape[2]
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    for l in range(n_layer):
        for h in range(n_head):
            q = all_q[l, h].float()   # (seq, d_head)
            k = all_k[l, h].float()

            # 3-bit MSE
            ind3_q, norm3_q = mse3_q.quantize(q)
            q_hat3 = mse3_q.dequantize(ind3_q, norm3_q).float()
            ind3_k, norm3_k = mse3_q.quantize(k)
            k_hat3 = mse3_q.dequantize(ind3_k, norm3_k).float()

            # 4-bit MSE
            ind4_q, norm4_q = mse4_q.quantize(q)
            q_hat4 = mse4_q.dequantize(ind4_q, norm4_q).float()
            ind4_k, norm4_k = mse4_q.quantize(k)
            k_hat4 = mse4_q.dequantize(ind4_k, norm4_k).float()

            # 4-bit QJL (3-bit MSE + 1-bit QJL)
            ind_qjl_q, norm_qjl_q, signs_qjl_q, gamma_qjl_q = qjl4_q.quantize(q)
            q_hat_qjl4 = qjl4_q.dequantize(ind_qjl_q, norm_qjl_q, signs_qjl_q, gamma_qjl_q).float()
            ind_qjl_k, norm_qjl_k, signs_qjl_k, gamma_qjl_k = qjl4_q.quantize(k)
            k_hat_qjl4 = qjl4_q.dequantize(ind_qjl_k, norm_qjl_k, signs_qjl_k, gamma_qjl_k).float()

            # Inner products (causal pairs)
            true_ips = (q @ k.T)[mask]
            ips_mse3 = (q_hat3 @ k_hat3.T)[mask]
            ips_mse4 = (q_hat4 @ k_hat4.T)[mask]
            ips_qjl4 = (q_hat_qjl4 @ k_hat_qjl4.T)[mask]

            mse3_acc[l, h, p_idx] = torch.mean((ips_mse3 - true_ips)**2).item()
            mse4_acc[l, h, p_idx] = torch.mean((ips_mse4 - true_ips)**2).item()
            qjl4_acc[l, h, p_idx] = torch.mean((ips_qjl4 - true_ips)**2).item()

            # Regression-based Bias-Variance decomposition
            sum_true_sq = torch.sum(true_ips**2).item()
            if sum_true_sq > 0:
                alpha4  = (torch.sum(ips_mse4 * true_ips) / sum_true_sq).item()
                bias4_acc[l, h, p_idx] = ((alpha4 - 1.0)**2 * torch.mean(true_ips**2)).item()
                var4_acc[l, h, p_idx]  = torch.mean((ips_mse4 - alpha4 * true_ips)**2).item()

                alphaq  = (torch.sum(ips_qjl4 * true_ips) / sum_true_sq).item()
                biasq_acc[l, h, p_idx] = ((alphaq - 1.0)**2 * torch.mean(true_ips**2)).item()
                varq_acc[l, h, p_idx]  = torch.mean((ips_qjl4 - alphaq * true_ips)**2).item()

# ---------------------------------------------------------------------------
# 7. Compute averaged statistics
# ---------------------------------------------------------------------------
mse3_mean   = mse3_acc.mean(axis=2)    # (n_layer, n_head)
mse4_mean   = mse4_acc.mean(axis=2)
qjl4_mean   = qjl4_acc.mean(axis=2)
mse3_std    = mse3_acc.std(axis=2)
mse4_std    = mse4_acc.std(axis=2)
qjl4_std    = qjl4_acc.std(axis=2)

# Win: QJL total MSE < MSE-4 total MSE (averaged across prompts)
qjl_better_matrix = qjl4_mean < mse4_mean   # (n_layer, n_head) bool

# Relative improvement per cell
rel_imp_matrix = np.where(
    mse4_mean > 0,
    (mse4_mean - qjl4_mean) / mse4_mean,
    0.0
)

# Summary statistics
qjl_better_count_per_layer = qjl_better_matrix.sum(axis=1)   # (n_layer,)
qjl_better_count_per_head  = qjl_better_matrix.sum(axis=0)   # (n_head,)
avg_qjl_better_heads = float(np.mean(qjl_better_count_per_layer))
total_wins = int(qjl_better_matrix.sum())
total_cells = n_layer * n_head

print(f"\n=== Empirical QJL Performance Summary at d={d_head} ({MODEL_NAME}) ===")
print(f"  Total wins  : {total_wins} / {total_cells} layer×head cells")
print(f"  Per-layer avg wins: {avg_qjl_better_heads:.2f} / {n_head} heads")
print(f"\nPer-head total wins across all layers:")
for h in range(n_head):
    bar = "█" * int(qjl_better_count_per_head[h])
    print(f"  Head {h:>2}: {qjl_better_count_per_head[h]:>3}/{n_layer}  {bar}")

# ---------------------------------------------------------------------------
# 8. Build layer_results list for JSON
# ---------------------------------------------------------------------------
layer_results = []
for l in range(n_layer):
    heads_data = []
    for h in range(n_head):
        heads_data.append({
            "head": h,
            "mse3": {"total_mse": float(mse3_mean[l, h]), "std": float(mse3_std[l, h])},
            "mse4": {
                "total_mse": float(mse4_mean[l, h]),
                "std":       float(mse4_std[l, h]),
                "sys_bias_sq": float(bias4_acc[l, h].mean()),
                "variance":    float(var4_acc[l, h].mean()),
            },
            "qjl4": {
                "total_mse": float(qjl4_mean[l, h]),
                "std":       float(qjl4_std[l, h]),
                "sys_bias_sq": float(biasq_acc[l, h].mean()),
                "variance":    float(varq_acc[l, h].mean()),
            },
            "qjl_better": bool(qjl_better_matrix[l, h]),
            "relative_improvement": float(rel_imp_matrix[l, h]),
        })
    layer_results.append({"layer": l, "heads": heads_data})

# ---------------------------------------------------------------------------
# 9. Save JSON
# ---------------------------------------------------------------------------
json_path = os.path.join(output_dir, "crossover_data_multiprompt.json")
summary_data = {
    "model": MODEL_NAME,
    "d_head": d_head,
    "n_layer": n_layer,
    "n_head": n_head,
    "n_prompts": len(PROMPTS),
    "total_wins": total_wins,
    "total_cells": total_cells,
    "avg_qjl_better_heads_per_layer": avg_qjl_better_heads,
    "qjl_better_count_per_head": qjl_better_count_per_head.tolist(),
    "qjl_better_count_per_layer": qjl_better_count_per_layer.tolist(),
    "layer_results": layer_results,
}
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2)
print(f"\nDetailed data saved to {json_path}")

# ---------------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

im = axes[0].imshow(rel_imp_matrix, aspect="auto", cmap="RdBu", vmin=-1.0, vmax=1.0)
axes[0].set_title(
    f"QJL Improvement over 4-bit MSE-only\n({MODEL_NAME}, {len(PROMPTS)} prompts avg)\n"
    "(Blue=better, Red=worse)",
    fontsize=12, fontweight="bold"
)
axes[0].set_xlabel("Attention Head", fontsize=11)
axes[0].set_ylabel("Transformer Layer", fontsize=11)
axes[0].set_xticks(range(n_head))
axes[0].set_yticks(range(n_layer))
fig.colorbar(im, ax=axes[0], label="Relative MSE Change: (MSE4 − QJL4) / MSE4")

axes[1].bar(range(n_head), qjl_better_count_per_head, color="#4a9eff", edgecolor="black", alpha=0.8)
axes[1].axhline(y=n_layer / 2, color="#ff5555", linestyle="--",
                label=f"50% Win Rate ({n_layer//2}/{n_layer} layers)")
axes[1].set_title(
    f"Layers where QJL Beats 4-bit MSE-only\n({MODEL_NAME}, d={d_head}, Total Layers={n_layer})",
    fontsize=12, fontweight="bold"
)
axes[1].set_xlabel("Attention Head", fontsize=11)
axes[1].set_ylabel("Win Count (Layers)", fontsize=11)
axes[1].set_xticks(range(n_head))
axes[1].set_ylim(0, n_layer + 1)
axes[1].grid(True, linestyle=":", alpha=0.5, axis="y")
axes[1].legend(fontsize=10)

plt.tight_layout()
plot_path = os.path.join(output_dir, "crossover_plot_multiprompt.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Performance breakdown plot saved to {plot_path}")
print("Multi-prompt evaluation completed successfully!")
