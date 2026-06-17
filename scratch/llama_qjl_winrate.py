"""
llama_qjl_winrate.py
====================
Per-layer, per-head QJL vs 4-bit MSE-only win-rate analysis on real
Llama-3.2 attention vectors.

Supported models:
  --model llama-1b  =>  meta-llama/Llama-3.2-1B-Instruct  (d_head=64,  16 layers, 32 Q-heads, 8 KV-heads)
  --model llama-3b  =>  meta-llama/Llama-3.2-3B-Instruct  (d_head=128, 28 layers, 24 Q-heads, 8 KV-heads)

Results saved to:
  results/llama_1b/winrate_data.json  (1B)
  results/llama_3b/winrate_data.json  (3B)

Usage:
  python scratch/llama_qjl_winrate.py --model llama-1b [--hf_token HF_TOKEN]
  python scratch/llama_qjl_winrate.py --model llama-3b [--hf_token HF_TOKEN]

Notes:
  - Requires HuggingFace token if running for the first time (gated repo).
  - Runs fully on CPU (no GPU required for analysis, but GPU accelerates forward pass).
  - Q vectors come from query heads (32 or 24); K vectors from KV heads (8).
    Win-rate is reported on KV-head K vectors paired with mean-pooled Q, and also
    on Q-head Q vectors for completeness. Main reported number uses KV heads.
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
parser = argparse.ArgumentParser(description="Llama QJL win-rate analysis")
parser.add_argument("--model", type=str, required=True,
                    choices=["llama-1b", "llama-3b"],
                    help="Which Llama model to analyse")
parser.add_argument("--hf_token", type=str, default=None,
                    help="HuggingFace access token (for gated models). "
                         "Also reads from HF_TOKEN env var.")
args = parser.parse_args()

MODEL_MAP = {
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
}
RESULTS_MAP = {
    "llama-1b": "llama_1b",
    "llama-3b": "llama_3b",
}

hf_model_id = MODEL_MAP[args.model]
results_subdir = RESULTS_MAP[args.model]

# HF token
hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)
if hf_token:
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)

# Seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Output directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(project_dir, "results", results_subdir)
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 4 diverse 128+ token prompts (plain text, no chat template needed for hooks)
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
# 1. Load model
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nLoading {hf_model_id} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if device == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    hf_model_id,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)
model.eval()

# Read architecture config
n_layer   = model.config.num_hidden_layers
n_q_heads = model.config.num_attention_heads
n_kv_heads = getattr(model.config, "num_key_value_heads", n_q_heads)
d_head    = model.config.hidden_size // n_q_heads
gqa_ratio = n_q_heads // n_kv_heads   # how many Q-heads share one KV-head

print(f"  Layers    : {n_layer}")
print(f"  Q-heads   : {n_q_heads}")
print(f"  KV-heads  : {n_kv_heads}")
print(f"  d_head    : {d_head}")
print(f"  GQA ratio : {gqa_ratio}:1")

# ---------------------------------------------------------------------------
# 2. Hook Llama attention layers to extract Q and K
#    LlamaAttention computes:
#      query_states = self.q_proj(hidden_states).view(bsz, q_len, n_q_heads, head_dim)
#      key_states   = self.k_proj(hidden_states).view(bsz, q_len, n_kv_heads, head_dim)
#    We hook q_proj and k_proj directly (pre-RoPE, raw projections).
#    RoPE changes direction but not magnitude — the quantization comparison is
#    valid on both pre- and post-RoPE vectors; we use pre-RoPE for simplicity.
# ---------------------------------------------------------------------------

def run_prompt_llama(prompt_text):
    """Extract Q and K projection outputs for all layers on a Llama model."""
    q_captures = {}   # layer_idx -> tensor (bsz, q_len, n_q_heads * d_head)
    k_captures = {}   # layer_idx -> tensor (bsz, q_len, n_kv_heads * d_head)

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

    # Shape q_raw: (bsz=1, seq, n_q_heads * d_head)
    # Shape k_raw: (bsz=1, seq, n_kv_heads * d_head)
    all_q_list = []
    all_k_list = []

    for li in range(n_layer):
        q_raw = q_captures[li][0].cpu()   # (seq, n_q_heads * d_head)
        k_raw = k_captures[li][0].cpu()   # (seq, n_kv_heads * d_head)

        seq_len = q_raw.shape[0]
        q_heads = q_raw.view(seq_len, n_q_heads, d_head).permute(1, 0, 2)   # (n_q_heads, seq, d_head)
        k_heads = k_raw.view(seq_len, n_kv_heads, d_head).permute(1, 0, 2)  # (n_kv_heads, seq, d_head)

        all_q_list.append(q_heads)
        all_k_list.append(k_heads)

    return all_q_list, all_k_list   # lists of length n_layer


# ---------------------------------------------------------------------------
# 3. Multi-prompt extraction
# ---------------------------------------------------------------------------
print(f"\nRunning {len(PROMPTS)} prompts through {hf_model_id}...")

per_prompt_q = []   # list[n_prompts] of list[n_layer] of (n_q_heads, seq, d_head)
per_prompt_k = []   # list[n_prompts] of list[n_layer] of (n_kv_heads, seq, d_head)

for idx, prompt in enumerate(PROMPTS):
    token_count = len(tokenizer.encode(prompt))
    print(f"  Prompt {idx+1}: {token_count} tokens")
    q_list, k_list = run_prompt_llama(prompt)
    per_prompt_q.append(q_list)
    per_prompt_k.append(k_list)

# ---------------------------------------------------------------------------
# 4. Print Q and K L2 norms per layer
# ---------------------------------------------------------------------------
print("\n--- Q and K L2 Norms per Layer (avg over tokens × heads × prompts) ---")
print(f"{'Layer':>5}  {'||Q|| mean':>12}  {'||Q|| std':>10}  {'||K|| mean':>12}  {'||K|| std':>10}")
for l in range(n_layer):
    q_norms, k_norms = [], []
    for p in range(len(PROMPTS)):
        q_p = per_prompt_q[p][l].float()   # (n_q_heads, seq, d_head)
        k_p = per_prompt_k[p][l].float()   # (n_kv_heads, seq, d_head)
        q_norms.append(torch.norm(q_p, dim=-1).reshape(-1))
        k_norms.append(torch.norm(k_p, dim=-1).reshape(-1))
    q_norms_all = torch.cat(q_norms)
    k_norms_all = torch.cat(k_norms)
    print(
        f"  {l:>3}  {q_norms_all.mean().item():>12.4f}  {q_norms_all.std().item():>10.4f}"
        f"  {k_norms_all.mean().item():>12.4f}  {k_norms_all.std().item():>10.4f}"
    )

# ---------------------------------------------------------------------------
# 5. Instantiate quantizers at d_head
# ---------------------------------------------------------------------------
mse4_q = TurboQuantMSE(d_head, bits=4, device="cpu", seed=SEED)
qjl4_q = TurboQuantProd(d_head, bits=4, device="cpu", seed=SEED)

# ---------------------------------------------------------------------------
# 6. Per-layer, per-KV-head win-rate (KV heads are the ones we quantize in cache)
# ---------------------------------------------------------------------------
print(f"\nEvaluating TurboQuantMSE & QJL on KV-head K vectors ({n_kv_heads} KV-heads per layer)...")

# Accumulators: (n_layer, n_kv_heads, n_prompts)
mse4_kv_acc  = np.zeros((n_layer, n_kv_heads, len(PROMPTS)))
qjl4_kv_acc  = np.zeros((n_layer, n_kv_heads, len(PROMPTS)))
bias4_kv_acc = np.zeros((n_layer, n_kv_heads, len(PROMPTS)))
var4_kv_acc  = np.zeros((n_layer, n_kv_heads, len(PROMPTS)))
biasq_kv_acc = np.zeros((n_layer, n_kv_heads, len(PROMPTS)))
varq_kv_acc  = np.zeros((n_layer, n_kv_heads, len(PROMPTS)))

for p_idx in range(len(PROMPTS)):
    for l in range(n_layer):
        # K vectors: (n_kv_heads, seq, d_head)
        k_all = per_prompt_k[p_idx][l].float()
        # Q vectors: (n_q_heads, seq, d_head)
        # For inner-product evaluation, expand KV heads to match Q heads (GQA broadcasting)
        q_all = per_prompt_q[p_idx][l].float()   # (n_q_heads, seq, d_head)

        seq_len = k_all.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        for kv_h in range(n_kv_heads):
            k = k_all[kv_h]   # (seq, d_head)

            # Corresponding Q heads for this KV head (GQA group)
            q_group_start = kv_h * gqa_ratio
            q_group = q_all[q_group_start : q_group_start + gqa_ratio]  # (gqa_ratio, seq, d_head)
            # Use the first Q head in the group (representative)
            q = q_group[0]    # (seq, d_head)

            # 4-bit MSE
            ind4_q, norm4_q = mse4_q.quantize(q)
            q_hat4 = mse4_q.dequantize(ind4_q, norm4_q).float()
            ind4_k, norm4_k = mse4_q.quantize(k)
            k_hat4 = mse4_q.dequantize(ind4_k, norm4_k).float()

            # 4-bit QJL
            ind_qjl_q, norm_qjl_q, signs_qjl_q, gamma_qjl_q = qjl4_q.quantize(q)
            q_hat_qjl4 = qjl4_q.dequantize(ind_qjl_q, norm_qjl_q, signs_qjl_q, gamma_qjl_q).float()
            ind_qjl_k, norm_qjl_k, signs_qjl_k, gamma_qjl_k = qjl4_q.quantize(k)
            k_hat_qjl4 = qjl4_q.dequantize(ind_qjl_k, norm_qjl_k, signs_qjl_k, gamma_qjl_k).float()

            true_ips = (q @ k.T)[mask]
            ips_mse4 = (q_hat4 @ k_hat4.T)[mask]
            ips_qjl4 = (q_hat_qjl4 @ k_hat_qjl4.T)[mask]

            mse4_kv_acc[l, kv_h, p_idx] = torch.mean((ips_mse4 - true_ips)**2).item()
            qjl4_kv_acc[l, kv_h, p_idx] = torch.mean((ips_qjl4 - true_ips)**2).item()

            sum_true_sq = torch.sum(true_ips**2).item()
            if sum_true_sq > 0:
                alpha4 = (torch.sum(ips_mse4 * true_ips) / sum_true_sq).item()
                bias4_kv_acc[l, kv_h, p_idx] = ((alpha4 - 1.0)**2 * torch.mean(true_ips**2)).item()
                var4_kv_acc[l, kv_h, p_idx]  = torch.mean((ips_mse4 - alpha4 * true_ips)**2).item()

                alphaq = (torch.sum(ips_qjl4 * true_ips) / sum_true_sq).item()
                biasq_kv_acc[l, kv_h, p_idx] = ((alphaq - 1.0)**2 * torch.mean(true_ips**2)).item()
                varq_kv_acc[l, kv_h, p_idx]  = torch.mean((ips_qjl4 - alphaq * true_ips)**2).item()

# ---------------------------------------------------------------------------
# 7. Average across prompts and compute win matrix
# ---------------------------------------------------------------------------
mse4_kv_mean = mse4_kv_acc.mean(axis=2)   # (n_layer, n_kv_heads)
qjl4_kv_mean = qjl4_kv_acc.mean(axis=2)

qjl_better_kv = qjl4_kv_mean < mse4_kv_mean   # (n_layer, n_kv_heads) bool
rel_imp_kv = np.where(
    mse4_kv_mean > 0,
    (mse4_kv_mean - qjl4_kv_mean) / mse4_kv_mean,
    0.0
)

wins_per_layer = qjl_better_kv.sum(axis=1)   # (n_layer,)
wins_per_head  = qjl_better_kv.sum(axis=0)   # (n_kv_heads,)
total_wins = int(qjl_better_kv.sum())
total_cells = n_layer * n_kv_heads
avg_wins_per_layer = float(wins_per_layer.mean())

print(f"\n=== Empirical QJL Win-Rate Summary ({hf_model_id}) ===")
print(f"  d_head      : {d_head}")
print(f"  KV-heads    : {n_kv_heads}")
print(f"  Layers      : {n_layer}")
print(f"  Prompts avg : {len(PROMPTS)}")
print(f"  Total wins  : {total_wins} / {total_cells} layer×KV-head cells")
print(f"  Avg KV-head wins per layer: {avg_wins_per_layer:.2f} / {n_kv_heads}")
print(f"\nPer-KV-head total wins across all layers:")
for h in range(n_kv_heads):
    bar = "█" * int(wins_per_head[h])
    print(f"  KV-Head {h}: {wins_per_head[h]:>3}/{n_layer}  {bar}")

# ---------------------------------------------------------------------------
# 8. Build layer_results for JSON
# ---------------------------------------------------------------------------
layer_results = []
for l in range(n_layer):
    heads_data = []
    for h in range(n_kv_heads):
        heads_data.append({
            "kv_head": h,
            "mse4": {
                "total_mse": float(mse4_kv_mean[l, h]),
                "std": float(mse4_kv_acc[l, h].std()),
                "sys_bias_sq": float(bias4_kv_acc[l, h].mean()),
                "variance": float(var4_kv_acc[l, h].mean()),
            },
            "qjl4": {
                "total_mse": float(qjl4_kv_mean[l, h]),
                "std": float(qjl4_kv_acc[l, h].std()),
                "sys_bias_sq": float(biasq_kv_acc[l, h].mean()),
                "variance": float(varq_kv_acc[l, h].mean()),
            },
            "qjl_better": bool(qjl_better_kv[l, h]),
            "relative_improvement": float(rel_imp_kv[l, h]),
        })
    layer_results.append({"layer": l, "heads": heads_data})

# ---------------------------------------------------------------------------
# 9. Save JSON
# ---------------------------------------------------------------------------
json_path = os.path.join(output_dir, "winrate_data.json")
summary = {
    "model": hf_model_id,
    "d_head": d_head,
    "n_layer": n_layer,
    "n_q_heads": n_q_heads,
    "n_kv_heads": n_kv_heads,
    "gqa_ratio": gqa_ratio,
    "n_prompts": len(PROMPTS),
    "total_wins": total_wins,
    "total_cells": total_cells,
    "avg_wins_per_layer": avg_wins_per_layer,
    "wins_per_layer": wins_per_layer.tolist(),
    "wins_per_head": wins_per_head.tolist(),
    "layer_results": layer_results,
}
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {json_path}")

# ---------------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n_layer // 3)))

im = axes[0].imshow(rel_imp_kv, aspect="auto", cmap="RdBu", vmin=-1.0, vmax=1.0)
axes[0].set_title(
    f"QJL vs 4-bit MSE: Relative Improvement\n({hf_model_id}, d_head={d_head})\n"
    "(Blue=QJL better, Red=MSE better)",
    fontsize=11, fontweight="bold"
)
axes[0].set_xlabel("KV Head", fontsize=10)
axes[0].set_ylabel("Layer", fontsize=10)
axes[0].set_xticks(range(n_kv_heads))
axes[0].set_yticks(range(n_layer))
fig.colorbar(im, ax=axes[0], label="(MSE4 − QJL4) / MSE4")

axes[1].bar(range(n_kv_heads), wins_per_head, color="#4a9eff", edgecolor="black", alpha=0.8)
axes[1].axhline(y=n_layer / 2, color="#ff5555", linestyle="--",
                label=f"50% ({n_layer//2}/{n_layer} layers)")
axes[1].set_title(
    f"Layers where QJL Beats 4-bit MSE\n({hf_model_id}, d_head={d_head})",
    fontsize=11, fontweight="bold"
)
axes[1].set_xlabel("KV Head", fontsize=10)
axes[1].set_ylabel("Win Count (Layers)", fontsize=10)
axes[1].set_xticks(range(n_kv_heads))
axes[1].set_ylim(0, n_layer + 1)
axes[1].grid(True, linestyle=":", alpha=0.5, axis="y")
axes[1].legend(fontsize=9)

plt.tight_layout()
plot_path = os.path.join(output_dir, "winrate_plot.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Win-rate plot saved to {plot_path}")
print(f"\nFinal answer: {total_wins}/{total_cells} KV-head×layer cells win at d_head={d_head}")
