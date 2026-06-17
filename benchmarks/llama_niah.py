"""
llama_niah.py
=============
Needle-In-A-Haystack (NIAH) baseline benchmark for Llama-3.2 models.

Runs the same NIAH task as mc_niah.py (same lengths, depths, trials, and
JSON schema) but evaluates only the FP16 baseline — no quantized memory
caching (MC-TurboQuant is GPT-2-specific architecture).

Models:
  --model llama-1b  =>  meta-llama/Llama-3.2-1B-Instruct
  --model llama-3b  =>  meta-llama/Llama-3.2-3B-Instruct

Output:
  results/llama_1b/niah.json   (1B)
  results/llama_3b/niah.json   (3B)

JSON schema:
  Same as mc_niah.json — dict of { arch_name: { L: { depth: accuracy } } }
  Only "baseline_fp16" key is populated for Llama.

Usage:
  python benchmarks/llama_niah.py --model llama-1b [--hf_token TOKEN]
  python benchmarks/llama_niah.py --model llama-3b [--hf_token TOKEN]
"""

import argparse
import os
import sys
import json
import random
import torch
import numpy as np
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Llama NIAH baseline benchmark")
parser.add_argument("--model", type=str, required=True,
                    choices=["llama-1b", "llama-3b"])
parser.add_argument("--hf_token", type=str, default=None)
args = parser.parse_args()

MODEL_MAP   = {"llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
               "llama-3b": "meta-llama/Llama-3.2-3B-Instruct"}
RESULTS_MAP = {"llama-1b": "llama_1b", "llama-3b": "llama_3b"}

hf_model_id  = MODEL_MAP[args.model]
results_subdir = RESULTS_MAP[args.model]

hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)
if hf_token:
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Load model + tokenizer
# ---------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer

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

max_pos = getattr(model.config, "max_position_embeddings", 2048)
print(f"  Max positions: {max_pos}")

# ---------------------------------------------------------------------------
# Synthetic NIAH prompt generator
# Wraps the secret in Llama-3.2 chat template so the model generates properly.
# ---------------------------------------------------------------------------
DISTRACTORS = [
    "The grass is green and the sky is blue. ",
    "Many people enjoy drinking hot coffee in the morning. ",
    "The sun rises in the east and sets in the west. ",
    "A standard laptop has a keyboard and a screen. ",
    "Python is a popular programming language for data science. ",
    "The capital of France is Paris, known for the Eiffel Tower. ",
    "Cats are popular pets known for their independence. ",
    "Water boils at 100 degrees Celsius under standard pressure. ",
]

def generate_niah_tokens(tokenizer, L: int, depth: float, secret_code: str) -> List[int]:
    """
    Build a sequence of exactly L tokens containing the secret code at
    relative depth `depth` in [0,1].
    """
    preamble = "There is a lot of information in this document. We have collected various facts for you. "
    fact = f"The secret number is {secret_code}. Remember this number. "
    question = "\nQuestion: What is the secret number?\nAnswer: The secret number is"

    preamble_ids = tokenizer.encode(preamble, add_special_tokens=False)
    fact_ids     = tokenizer.encode(fact,     add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)

    needed_dist = L - len(preamble_ids) - len(fact_ids) - len(question_ids)
    if needed_dist < 0:
        needed_dist = 0

    dist_ids: List[int] = []
    i = 0
    while len(dist_ids) < needed_dist:
        dist_ids.extend(tokenizer.encode(DISTRACTORS[i % len(DISTRACTORS)],
                                         add_special_tokens=False))
        i += 1
    dist_ids = dist_ids[:needed_dist]

    split = int(depth * len(dist_ids))
    dist_pre  = dist_ids[:split]
    dist_post = dist_ids[split:]

    full = preamble_ids + dist_pre + fact_ids + dist_post + question_ids

    # Trim or pad to exactly L tokens
    if len(full) > L:
        full = full[:L]
    return full


# ---------------------------------------------------------------------------
# NIAH evaluation loop
# ---------------------------------------------------------------------------
lengths  = [256, 512, 768, 1024]
depths   = {"Early": 0.1, "Middle": 0.5, "Late": 0.9}
n_trials = 10

# Keep only lengths within model's max context
lengths = [l for l in lengths if l <= max_pos]

print(f"\nRunning NIAH: lengths={lengths}, depths={list(depths.keys())}, trials={n_trials}")

matrix = {l: {d: 0.0 for d in depths} for l in lengths}

for L in lengths:
    for d_name, d_val in depths.items():
        successes = 0
        for trial in range(n_trials):
            random.seed(42 + L + int(d_val * 100) + trial)
            secret_code = f"{random.randint(1000, 9999)}"

            prompt_ids = generate_niah_tokens(tokenizer, L - 4, d_val, secret_code)
            input_ids  = torch.tensor([prompt_ids], device=device)

            with torch.no_grad():
                outputs    = model(input_ids, use_cache=False)
                logits     = outputs.logits
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated  = [next_token.item()]

                # Generate 3 more tokens (total 4 to cover 4-digit codes)
                for step in range(3):
                    new_ids = torch.cat([input_ids,
                                         torch.tensor([generated], device=device)], dim=-1)
                    out2 = model(new_ids, use_cache=False)
                    nt   = out2.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated.append(nt.item())

            gen_text = tokenizer.decode(generated, skip_special_tokens=True)
            if secret_code in gen_text:
                successes += 1

        acc = (successes / n_trials) * 100.0
        matrix[L][d_name] = acc
        print(f"  L={L}, Depth={d_name} -> {acc:.0f}%  (secret found {successes}/{n_trials})")

# ---------------------------------------------------------------------------
# Save JSON — same schema as mc_niah.json
# ---------------------------------------------------------------------------
results = {"baseline_fp16": matrix}

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(project_dir, "results", results_subdir)
os.makedirs(out_dir, exist_ok=True)

json_path = os.path.join(out_dir, "niah.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {json_path}")
print("NIAH evaluation complete.")
