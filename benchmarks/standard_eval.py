"""
standard_eval.py
================
Runs three industry-standard evaluations:
  - Eval A: WikiText-2 Perplexity (strided sliding window, stride=512, context=1024)
  - Eval B: HellaSwag Accuracy (100 validation examples scored by log-likelihood)
  - Eval C: Long-context Needle-In-A-Haystack (lengths [256, 512, 768, 1024], depths Early/Middle/Late, 10 trials each)

Produces:
  - results/standard_eval.json
  - results/standard_eval.html
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc
import json
import math
import time
import random
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# Configuration
SEED = 42
MODEL_NAME = "gpt2-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ----------------------------------------------------------------------
# Quantized Attention Forward Patches (Dynamic Activation Quantization)
# ----------------------------------------------------------------------
def make_quantized_attn_forward(quant_scheme: Optional[str], bits: int, device: str):
    """
    Returns a forward pass function that applies key-value activation quantization.
    - 'turboquant': Original QR rotation + symmetric QJL (reimported from turbo_quant_demo.py)
    - 'wht_quantizer': WHT rotation + asymmetric QJL (reimported from wht_quantizer.py)
    """
    if quant_scheme is None:
        return None

    # Lazy import to avoid circular dependencies
    from turboquant.turbo_quant_demo import TurboQuantProd as OrigTurboQuantProd
    from turboquant.wht_quantizer import TurboQuantProd as WhtTurboQuantProd
    from turboquant.wht_quantizer import TurboQuantMSE as WhtTurboQuantMSE

    num_layers = 24
    d_head = 64
    k_quantizers = []
    v_quantizers = []

    for li in range(num_layers):
        if quant_scheme == "turboquant":
            # Symmetric: Prod on both Key and Value
            k_quantizers.append(OrigTurboQuantProd(d_head, bits, device, seed=SEED + li * 100))
            v_quantizers.append(OrigTurboQuantProd(d_head, bits, device, seed=SEED + li * 100 + 50))
        elif quant_scheme == "wht_quantizer":
            # Asymmetric: Prod on Key, MSE on Value
            k_quantizers.append(WhtTurboQuantProd(d_head, bits, device, seed=SEED + li * 100))
            v_quantizers.append(WhtTurboQuantMSE(d_head, bits, device, seed=SEED + li * 100 + 50))
        else:
            k_quantizers.append(None)
            v_quantizers.append(None)

    def quantized_forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, **kwargs):
        is_cross_attention = kwargs.get("encoder_hidden_states", None) is not None
        if is_cross_attention:
            return self._orig_forward(hidden_states, past_key_value, cache_position, attention_mask, **kwargs)

        # 1. Compute Q, K, V
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        layer_idx = self.layer_idx
        k_quantizer = k_quantizers[layer_idx]
        v_quantizer = v_quantizers[layer_idx]

        if k_quantizer is not None and v_quantizer is not None:
            # Key: quantize & dequantize new tokens
            k_quant = k_quantizer.quantize(key_states)
            key_states = k_quantizer.dequantize(*k_quant)

            # Value: quantize & dequantize new tokens
            v_quant = v_quantizer.quantize(value_states)
            value_states = v_quantizer.dequantize(*v_quant)

        if past_key_value is not None:
            if isinstance(past_key_value, tuple):
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            else:
                key_states, value_states = past_key_value.update(key_states, value_states, layer_idx)

        # Standard causal scaled dot-product attention calculation
        q_len = query_states.shape[2]
        k_len = key_states.shape[2]
        attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            bias = torch.tril(torch.ones(q_len, k_len, device=query_states.device)).view(1, 1, q_len, k_len)
            attn_weights = attn_weights.masked_fill(bias == 0, float('-inf'))
            
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, value_states.float())
        attn_output = attn_output.to(query_states.dtype)

        # Format output back
        attn_output = attn_output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        present = past_key_value if past_key_value is not None and not isinstance(past_key_value, tuple) else (key_states, value_states)
        return attn_output, present

    return quantized_forward


def patch_attention(model, quant_scheme: Optional[str], bits: int):
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    
    # Store original forward if not already done
    if not hasattr(GPT2Attention, "_orig_forward"):
        GPT2Attention._orig_forward = GPT2Attention.forward

    # Ensure layer_idx is set on all layers
    for i, block in enumerate(model.transformer.h):
        block.attn.layer_idx = i

    if quant_scheme is None:
        GPT2Attention.forward = GPT2Attention._orig_forward
    else:
        q_forward = make_quantized_attn_forward(quant_scheme, bits, DEVICE)
        GPT2Attention.forward = lambda self, *args, **kwargs: q_forward(self, *args, **kwargs)


def unpatch_attention():
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    if hasattr(GPT2Attention, "_orig_forward"):
        GPT2Attention.forward = GPT2Attention._orig_forward


# ----------------------------------------------------------------------
# Eval A: WikiText-2 Perplexity (Strided Sliding Window)
# ----------------------------------------------------------------------
def eval_wikitext_perplexity(model, tokenizer, max_tokens: int = 35000) -> float:
    print(f"Loading WikiText-2 test split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    
    seq_len = encodings.input_ids.size(1)
    eval_len = min(seq_len, max_tokens)
    print(f"Evaluating PPL on first {eval_len} tokens (sliding window stride=512, ctx=1024)...")
    
    max_length = 1024
    stride = 512
    nlls = []
    
    prev_end_loc = 0
    for begin_loc in range(0, eval_len, stride):
        end_loc = min(begin_loc + max_length, eval_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss.float() * trg_len
            
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == eval_len:
            break
            
    perplexity = math.exp(torch.stack(nlls).sum().item() / eval_len)
    return perplexity


# ----------------------------------------------------------------------
# Eval B: HellaSwag Accuracy (100 validation examples)
# ----------------------------------------------------------------------
def score_hellaswag_example(model, tokenizer, ctx: str, ending: str) -> float:
    ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
    full_prompt = ctx + " " + ending
    input_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    
    if len(input_ids) <= len(ctx_ids):
        return -9999.0
        
    input_tensor = torch.tensor([input_ids], device=DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits
        shift_logits = logits[0, :-1, :]
        shift_labels = input_tensor[0, 1:]
        
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        ending_log_prob = token_log_probs[len(ctx_ids) - 1:].sum().item()
        
    return ending_log_prob


def eval_hellaswag(model, tokenizer, num_examples: int = 100) -> float:
    print(f"Loading HellaSwag validation split...")
    dataset = load_dataset("hellaswag", split="validation")
    
    correct = 0
    total = 0
    
    for idx in range(num_examples):
        example = dataset[idx]
        ctx = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])
        
        scores = []
        for ending in endings:
            scores.append(score_hellaswag_example(model, tokenizer, ctx, ending))
            
        predicted = np.argmax(scores)
        if predicted == label:
            correct += 1
        total += 1
        
    accuracy = (correct / total) * 100.0
    return accuracy


# ----------------------------------------------------------------------
# Eval C: Long-Context NIAH Matrix Benchmark
# ----------------------------------------------------------------------
class SegmentedKVCacheHook:
    """MC Segmented Cache Hook for MC-TurboQuant."""
    def __init__(self, num_layers: int, d_head: int, bits: int, device: str, compressed: bool):
        from benchmarks.mc_niah import LayerMCCache
        self.caches = {}
        for li in range(num_layers):
            self.caches[li] = LayerMCCache(li, d_head, bits, device, compressed)
            
    def clear(self):
        for li in self.caches:
            self.caches[li].clear()


def make_mc_attention_forward(cache_hook: SegmentedKVCacheHook):
    def mc_forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, **kwargs):
        is_cross_attention = kwargs.get("encoder_hidden_states", None) is not None
        if is_cross_attention:
            return self._orig_forward(hidden_states, past_key_value, cache_position, attention_mask, **kwargs)

        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        # Update layer's segmented cache
        layer_cache = cache_hook.caches[self.layer_idx]
        layer_cache.update(key_states, value_states)

        # Reconstruct segments
        segments = []
        for item in layer_cache.completed:
            if layer_cache.compressed:
                k_seg = layer_cache.key_quantizer.dequantize(*item[0])
                v_seg = layer_cache.val_quantizer.dequantize(*item[1])
            else:
                k_seg, v_seg = item
            segments.append((k_seg, v_seg))

        if layer_cache.curr_k is not None and layer_cache.curr_k.shape[2] > 0:
            segments.append((layer_cache.curr_k, layer_cache.curr_v))

        # LogSumExp Gated Residual Attention
        outputs_list = []
        r_list = []
        head_dim = query_states.size(-1)
        q_len = query_states.shape[2]
        total_seq_len = sum(k_seg.shape[2] for k_seg, _ in segments)
        q_start = total_seq_len - q_len

        q_32 = query_states.float()
        k_start = 0
        for k_seg, v_seg in segments:
            k_len = k_seg.shape[2]
            scores = torch.matmul(q_32, k_seg.float().transpose(-1, -2)) / math.sqrt(head_dim)
            
            t_indices = torch.arange(q_len, device=query_states.device).unsqueeze(1) + q_start
            j_indices = torch.arange(k_len, device=query_states.device).unsqueeze(0) + k_start
            mask = j_indices > t_indices
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            # Prevent NaNs
            all_masked = mask.all(dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            safe_scores = scores.masked_fill(all_masked, 0.0)
            probs = torch.softmax(safe_scores, dim=-1).masked_fill(all_masked, 0.0)

            outputs_list.append(torch.matmul(probs, v_seg.float()))
            r_list.append(torch.logsumexp(scores, dim=-1))
            k_start += k_len

        r = torch.stack(r_list, dim=-1)
        gamma = torch.softmax(r, dim=-1)

        attn_output = torch.zeros_like(q_32)
        for i, output_i in enumerate(outputs_list):
            attn_output += gamma[..., i].unsqueeze(-1) * output_i

        attn_output = attn_output.to(query_states.dtype)
        attn_output = attn_output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, None

    return mc_forward


def generate_niah_prompt_with_depth(tokenizer, L: int, depth: float, secret_code: str) -> List[int]:
    preamble = "There is a lot of information in this document. We have collected various facts for you. "
    fact = f"The secret number is {secret_code}. Remember this number. "
    question = "\nQuestion: What is the secret number?\nAnswer: The secret number is"

    preamble_ids = tokenizer.encode(preamble)
    fact_ids = tokenizer.encode(fact)
    question_ids = tokenizer.encode(question)

    needed_dist_len = L - len(preamble_ids) - len(fact_ids) - len(question_ids)
    if needed_dist_len < 0:
        raise ValueError(f"Target length L={L} is too short.")

    distractors = [
        "The grass is green and the sky is blue. ",
        "Many people enjoy drinking hot coffee in the morning. ",
        "The sun rises in the east and sets in the west. ",
        "A standard laptop has a keyboard and a screen. ",
        "Python is a popular programming language for data science. ",
        "The capital of France is Paris, known for the Eiffel Tower. ",
        "Cats are popular pets known for their independence. ",
        "Water boils at 100 degrees Celsius under standard pressure. ",
    ]

    dist_ids = []
    i = 0
    while len(dist_ids) < needed_dist_len:
        ids = tokenizer.encode(distractors[i % len(distractors)])
        dist_ids.extend(ids)
        i += 1
    dist_ids = dist_ids[:needed_dist_len]

    # Split distractors at depth
    split_idx = int(depth * len(dist_ids))
    dist_pre = dist_ids[:split_idx]
    dist_post = dist_ids[split_idx:]

    full_ids = preamble_ids + dist_pre + fact_ids + dist_post + question_ids
    assert len(full_ids) == L
    return full_ids


def eval_niah_architecture(model, tokenizer, arch_type: str) -> Dict[str, Dict[str, float]]:
    lengths = [256, 512, 768, 1024]
    depths = {"Early": 0.1, "Middle": 0.5, "Late": 0.9}
    n_trials = 10
    
    # Store matrix results: length -> depth -> success_pct
    matrix = {l: {d: 0.0 for d in depths} for l in lengths}
    
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    
    # Configure patching/cache based on architecture
    cache_hook = None
    if arch_type == "mc_turboquant":
        num_layers = model.config.n_layer
        d_head = model.config.n_embd // model.config.n_head
        cache_hook = SegmentedKVCacheHook(num_layers, d_head, bits=4, device=DEVICE, compressed=True)
        mc_forward = make_mc_attention_forward(cache_hook)
        GPT2Attention.forward = lambda self, *args, **kwargs: mc_forward(self, *args, **kwargs)
    elif arch_type == "turboquant_flat":
        patch_attention(model, "turboquant", 4)
    elif arch_type == "wht_asym_flat":
        patch_attention(model, "wht_quantizer", 4)
    else:
        # Baseline FP16
        unpatch_attention()
        
    for L in lengths:
        for depth_name, depth_val in depths.items():
            successes = 0
            for trial in range(n_trials):
                # Unique seed based on params for reproducibility but trial-level randomness
                random.seed(42 + L + int(depth_val * 100) + trial)
                secret_code = f"{random.randint(1000, 9999)}"
                
                prompt_ids = generate_niah_prompt_with_depth(tokenizer, L - 4, depth_val, secret_code)
                input_ids = torch.tensor([prompt_ids], device=DEVICE)
                
                if cache_hook is not None:
                    cache_hook.clear()
                    
                with torch.no_grad():
                    # Process prompt (prefill)
                    if cache_hook is not None:
                        outputs = model(input_ids, use_cache=False)
                    else:
                        # For flat models, we patch attention globally so they run quantized forwards directly
                        outputs = model(input_ids, use_cache=False)
                    
                    logits = outputs.logits
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = [next_token.item()]
                    
                    # Generate 4 tokens
                    for step in range(3):
                        if cache_hook is not None:
                            pos = torch.tensor([[L - 4 + step]], device=DEVICE)
                            outputs = model(next_token, position_ids=pos, use_cache=False)
                        else:
                            # For flat, we evaluate without HuggingFace cache but using patched attention
                            # which means we re-run the full sequence, but to be fair we offset position IDs
                            # wait, running without cache means we just append the generated token and run full forward:
                            input_ids = torch.cat([input_ids, next_token], dim=-1)
                            outputs = model(input_ids, use_cache=False)
                            
                        logits = outputs.logits
                        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        generated.append(next_token.item())
                        
                gen_text = tokenizer.decode(generated, skip_special_tokens=True)
                if secret_code in gen_text:
                    successes += 1
                    
            matrix[L][depth_name] = (successes / n_trials) * 100.0
            print(f"  [{arch_type}] L={L}, Depth={depth_name} -> {matrix[L][depth_name]:.0f}%")
            
    # Cleanup patching
    unpatch_attention()
    return matrix


# ----------------------------------------------------------------------
# Main Benchmarking Orchestrator
# ----------------------------------------------------------------------
def main():
    if "--longbench-only" in sys.argv:
        run_longbench_lite()
        return

    print("=" * 70)
    print("  kvquant-lab: Industry-Standard Evaluation Suite")
    print("=" * 70)
    
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 1. Warmup
    print("Warming up model...")
    warmup_ids = tokenizer.encode("Hello quantum computer", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model(warmup_ids)
    torch.cuda.synchronize()
    
    results = {}
    
    # --- EVAL A & B ---
    configs = [
        {"name": "Baseline FP16", "scheme": None, "bits": 16},
        {"name": "Original TurboQuant (3-bit)", "scheme": "turboquant", "bits": 3},
        {"name": "Original TurboQuant (4-bit)", "scheme": "turboquant", "bits": 4},
        {"name": "WHT + Asymmetric QJL (3-bit)", "scheme": "wht_quantizer", "bits": 3},
        {"name": "WHT + Asymmetric QJL (4-bit)", "scheme": "wht_quantizer", "bits": 4},
    ]
    
    eval_a_ppls = {}
    eval_b_accs = {}
    
    for cfg in configs:
        name = cfg["name"]
        print(f"\nEvaluating configuration: {name}")
        
        # Patch attention
        patch_attention(model, cfg["scheme"], cfg["bits"])
        
        # Run WikiText-2 slide perplexity (first 25,000 tokens for speed)
        ppl = eval_wikitext_perplexity(model, tokenizer, max_tokens=25000)
        print(f"  WikiText-2 Perplexity: {ppl:.4f}")
        eval_a_ppls[name] = ppl
        
        # Run HellaSwag Accuracy
        hs_acc = eval_hellaswag(model, tokenizer, num_examples=100)
        print(f"  HellaSwag Accuracy (100 examples): {hs_acc:.2f}%")
        eval_b_accs[name] = hs_acc
        
        # Unpatch attention
        unpatch_attention()
        
    results["eval_a_perplexity"] = eval_a_ppls
    results["eval_b_hellaswag"] = eval_b_accs
    
    # --- EVAL C (NIAH) ---
    print("\nEvaluating Objective 1 - Eval C: Long-Context NIAH Matrix Benchmark")
    architectures = ["baseline_fp16", "turboquant_flat", "wht_asym_flat", "mc_turboquant"]
    niah_matrices = {}
    
    for arch in architectures:
        print(f"\nRunning NIAH for: {arch}")
        matrix = eval_niah_architecture(model, tokenizer, arch)
        niah_matrices[arch] = matrix
        
    results["eval_c_niah"] = niah_matrices
    
    # Save JSON results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    json_path = os.path.join(results_dir, "standard_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to {json_path}")
    
    # Generate HTML report
    html_path = os.path.join(results_dir, "standard_eval.html")
    generate_html(results, html_path)
    print(f"HTML report saved to {html_path}")
    
    # Run longbench-lite evaluations
    run_longbench_lite()


def generate_html(results: Dict, output_path: str):
    ppl = results["eval_a_perplexity"]
    hs = results["eval_b_hellaswag"]
    niah = results["eval_c_niah"]
    
    # Format tables for PPL and HellaSwag
    eval_a_rows = ""
    for name in ppl:
        eval_a_rows += f"""
        <tr>
            <td style="color:#ffffff;font-weight:600;">{name}</td>
            <td class="num-val highlight">{ppl[name]:.4f}</td>
            <td class="num-val">{hs[name]:.2f}%</td>
        </tr>
        """
        
    # Format NIAH matrices
    niah_html = ""
    arch_labels = {
        "baseline_fp16": "Baseline FP16",
        "turboquant_flat": "Original TurboQuant Flat (4-bit)",
        "wht_asym_flat": "WHT + Asymmetric QJL Flat (4-bit)",
        "mc_turboquant": "MC-TurboQuant (64-seg, WHT-compressed, 4-bit)"
    }
    
    for arch in ["baseline_fp16", "turboquant_flat", "wht_asym_flat", "mc_turboquant"]:
        matrix = niah[arch]
        matrix_rows = ""
        for L in ["256", "512", "768", "1024"]:
            l_int = int(L)
            matrix_rows += f"""
            <tr>
                <td class="dim-val">{L}</td>
                <td class="num-val">{(matrix[l_int]['Early'] if L in matrix or l_int in matrix else 0.0):.0f}%</td>
                <td class="num-val">{(matrix[l_int]['Middle'] if L in matrix or l_int in matrix else 0.0):.0f}%</td>
                <td class="num-val">{(matrix[l_int]['Late'] if L in matrix or l_int in matrix else 0.0):.0f}%</td>
            </tr>
            """
        
        niah_html += f"""
        <div class="card">
            <h3>{arch_labels[arch]}</h3>
            <table>
                <thead>
                    <tr>
                        <th>Length (L)</th>
                        <th class="num-val">Early (10%)</th>
                        <th class="num-val">Middle (50%)</th>
                        <th class="num-val">Late (90%)</th>
                    </tr>
                </thead>
                <tbody>
                    {matrix_rows}
                </tbody>
            </table>
        </div>
        """
        
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>kvquant-lab: Standard Evaluation Suite</title>
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
  .card h3 {{
    font-size: 14px;
    font-weight: 600;
    color: #4a9eff;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
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
  .niah-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
    gap: 20px;
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
    <h1>kvquant-lab: Standard Evaluation Suite Dashboard</h1>
    <div class="subtitle">WikiText-2 Strided sliding window perplexity &middot; HellaSwag log-likelihood accuracy &middot; Long-context Needle-In-A-Haystack Matrix</div>
  </header>

  <div class="card">
    <h2>Eval A &amp; B: Language Modeling and Downstream Task Accuracy</h2>
    <table>
      <thead>
        <tr>
          <th>Quantization Configuration</th>
          <th class="num-val">WikiText-2 Perplexity (1024-ctx, 512-stride)</th>
          <th class="num-val">HellaSwag Accuracy (%)</th>
        </tr>
      </thead>
      <tbody>
        {eval_a_rows}
      </tbody>
    </table>
  </div>

  <h2>Eval C: Long-Context Needle-In-A-Haystack Recall Accuracy Matrices</h2>
  <div class="niah-grid">
    {niah_html}
  </div>

  <div class="footer">
    Generated by standard_eval.py &middot; GPT-2 Medium &middot; Device: {DEVICE.upper()}
  </div>
</div>
</body>
</html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ----------------------------------------------------------------------
# GAP 2: LongBench-style Evaluation (Adapted for GPT-2)
# ----------------------------------------------------------------------
def patch_outlier_attention(model, avg_bits: float):
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    from turboquant.outlier_channel_quantizer import OutlierChannelQuantizer
    
    if not hasattr(GPT2Attention, "_orig_forward"):
        GPT2Attention._orig_forward = GPT2Attention.forward

    num_layers = model.config.n_layer
    d_head = model.config.n_embd // model.config.n_head
    k_quantizers = [OutlierChannelQuantizer(d_head, avg_bits, DEVICE, seed=SEED + li * 100) for li in range(num_layers)]
    v_quantizers = [OutlierChannelQuantizer(d_head, avg_bits, DEVICE, seed=SEED + li * 100 + 50) for li in range(num_layers)]
    
    def outlier_forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, **kwargs):
        is_cross_attention = kwargs.get("encoder_hidden_states", None) is not None
        if is_cross_attention:
            return self._orig_forward(hidden_states, past_key_value, cache_position, attention_mask, **kwargs)

        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        layer_idx = self.layer_idx
        k_comp = k_quantizers[layer_idx].quantize(key_states)
        key_states = k_quantizers[layer_idx].dequantize(k_comp)

        v_comp = v_quantizers[layer_idx].quantize(value_states)
        value_states = v_quantizers[layer_idx].dequantize(v_comp)

        if past_key_value is not None:
            if isinstance(past_key_value, tuple):
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            else:
                key_states, value_states = past_key_value.update(key_states, value_states, layer_idx)

        q_len = query_states.shape[2]
        k_len = key_states.shape[2]
        attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            bias = torch.tril(torch.ones(q_len, k_len, device=query_states.device)).view(1, 1, q_len, k_len)
            attn_weights = attn_weights.masked_fill(bias == 0, float('-inf'))
            
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, value_states.float())
        attn_output = attn_output.to(query_states.dtype)

        attn_output = attn_output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        present = past_key_value if past_key_value is not None and not isinstance(past_key_value, tuple) else (key_states, value_states)
        return attn_output, present

    GPT2Attention.forward = outlier_forward
    for i, block in enumerate(model.transformer.h):
        block.attn.layer_idx = i


def run_longbench_lite():
    print("=" * 80)
    print("Running LongBench-lite evaluations (adapted for GPT-2)...")
    print("=" * 80)
    
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    
    schemes = ["Baseline", "WHT+Asym-4bit", "Outlier-2.5bit", "Outlier-3.5bit"]
    results = {sch: {"passage_qa": 0.0, "summarization": 0.0, "code_completion": 0.0} for sch in schemes}
    
    def patch_model_scheme(scheme):
        unpatch_attention()
        if scheme == "Baseline":
            pass
        elif scheme == "WHT+Asym-4bit":
            patch_attention(model, "wht_quantizer", 4)
        elif scheme == "Outlier-2.5bit":
            patch_outlier_attention(model, 2.5)
        elif scheme == "Outlier-3.5bit":
            patch_outlier_attention(model, 3.5)
            
    # Task 1: PassageQA
    print("\n[Task 1/3] PassageQA (TriviaQA RC Validation, 20 examples)...")
    trivia_ds = load_dataset("trivia_qa", "rc", split="validation", streaming=True)
    trivia_examples = []
    
    def get_trivia_qa_context(ex) -> str:
        if ex.get("entity_pages") and ex["entity_pages"].get("wiki_context"):
            texts = [t for t in ex["entity_pages"]["wiki_context"] if t]
            if texts:
                return "\n".join(texts)
        if ex.get("search_results") and ex["search_results"].get("search_context"):
            texts = [t for t in ex["search_results"]["search_context"] if t]
            if texts:
                return "\n".join(texts)
        return ""
        
    for ex in trivia_ds:
        ctx = get_trivia_qa_context(ex)
        if ctx and len(ex["question"]) > 0 and len(ex["answer"]["aliases"]) > 0:
            trivia_examples.append((ctx, ex["question"], [a.lower().strip() for a in ex["answer"]["aliases"]]))
            if len(trivia_examples) >= 20:
                break
                
    for sch in schemes:
        patch_model_scheme(sch)
        correct = 0
        for ctx, q, aliases in trivia_examples:
            ctx_tokens = tokenizer.encode(ctx)[:900]
            q_str = f"\nQuestion: {q}\nAnswer:"
            q_tokens = tokenizer.encode(q_str)
            input_tokens = ctx_tokens + q_tokens
            input_tensor = torch.tensor([input_tokens], device=DEVICE)
            
            with torch.no_grad():
                generated_ids = model.generate(input_tensor, max_new_tokens=50, use_cache=True, pad_token_id=tokenizer.eos_token_id)
                        
            gen_text = tokenizer.decode(generated_ids[0][len(input_tokens):]).lower().strip()
            matched = any(alias in gen_text for alias in aliases)
            if matched:
                correct += 1
        acc = (correct / len(trivia_examples)) * 100.0
        results[sch]["passage_qa"] = acc
        print(f"  Scheme: {sch:16s} | PassageQA Acc: {acc:.1f}%")
        
    # Task 2: Summarization proxy
    print("\n[Task 2/3] Summarization Proxy (CNN/DailyMail Validation, 20 examples)...")
    cnn_ds = load_dataset("cnn_dailymail", "3.0.0", split="validation", streaming=True)
    cnn_examples = []
    for ex in cnn_ds:
        if ex.get("article") and ex.get("highlights"):
            cnn_examples.append((ex["article"], ex["highlights"]))
            if len(cnn_examples) >= 20:
                break
                
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    
    for sch in schemes:
        patch_model_scheme(sch)
        scores = []
        for article, ref_summary in cnn_examples:
            art_tokens = tokenizer.encode(article)[:800]
            prompt_str = "\nSummarize: "
            p_tokens = tokenizer.encode(prompt_str)
            input_tokens = art_tokens + p_tokens
            input_tensor = torch.tensor([input_tokens], device=DEVICE)
            
            with torch.no_grad():
                generated_ids = model.generate(input_tensor, max_new_tokens=100, use_cache=True, pad_token_id=tokenizer.eos_token_id)
                        
            gen_text = tokenizer.decode(generated_ids[0][len(input_tokens):]).strip()
            score = scorer.score(ref_summary, gen_text)["rouge1"].fmeasure * 100.0
            scores.append(score)
        avg_rouge = float(np.mean(scores))
        results[sch]["summarization"] = avg_rouge
        print(f"  Scheme: {sch:16s} | ROUGE-1 Score: {avg_rouge:.2f}")

    # Task 3: Code completion
    print("\n[Task 3/3] Code Completion (CodeSearchNet Python split, 20 examples)...")
    code_ds = load_dataset("code_search_net", "python", split="validation", streaming=True)
    code_examples = []
    for ex in code_ds:
        if ex.get("func_code_string"):
            code_examples.append(ex["func_code_string"])
            if len(code_examples) >= 20:
                break
                
    for sch in schemes:
        patch_model_scheme(sch)
        correct = 0
        for code in code_examples:
            full_tokens = tokenizer.encode(code)
            if len(full_tokens) < 100:
                prompt_tokens = full_tokens[:len(full_tokens)//2]
                ref_tokens = full_tokens[len(full_tokens)//2:]
            else:
                prompt_tokens = full_tokens[:50]
                ref_tokens = full_tokens[50:100]
                
            input_tensor = torch.tensor([prompt_tokens], device=DEVICE)
            
            with torch.no_grad():
                generated_ids = model.generate(input_tensor, max_new_tokens=50, use_cache=True, pad_token_id=tokenizer.eos_token_id)
                        
            gen_text = tokenizer.decode(generated_ids[0][len(prompt_tokens):])
            ref_text = tokenizer.decode(ref_tokens)
            
            gen_lines = [l.strip() for l in gen_text.split("\n") if l.strip()]
            ref_lines = [l.strip() for l in ref_text.split("\n") if l.strip()]
            
            if gen_lines and ref_lines and gen_lines[0] == ref_lines[0]:
                correct += 1
                
        acc = (correct / len(code_examples)) * 100.0
        results[sch]["code_completion"] = acc
        print(f"  Scheme: {sch:16s} | Code Completion Acc: {acc:.1f}%")
        
    unpatch_attention()
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "longbench_lite.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved longbench_lite results to {json_path}")
    
    # Generate HTML report
    html_path = os.path.join(results_dir, "longbench_lite.html")
    generate_longbench_html(results, html_path)
    print(f"Saved HTML report to {html_path}")


def generate_longbench_html(results, output_path):
    rows = ""
    for sch in ["Baseline", "WHT+Asym-4bit", "Outlier-2.5bit", "Outlier-3.5bit"]:
        r = results[sch]
        rows += f"""
        <tr>
            <td style="color:#8be9fd; font-weight:600;">{sch}</td>
            <td class="num-val highlight">{r['passage_qa']:.1f}%</td>
            <td class="num-val highlight">{r['summarization']:.2f}</td>
            <td class="num-val highlight">{r['code_completion']:.1f}%</td>
        </tr>
        """
        
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LongBench-lite Evaluation Dashboard</title>
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
    max-width: 900px;
    margin: 0 auto;
  }}
  header {{
    text-align: center;
    margin-bottom: 40px;
  }}
  header h1 {{
    font-size: 26px;
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
    <h1>LongBench-lite Evaluation Dashboard</h1>
    <div class="subtitle">Empirical Comparison comparable to Table 4 of the TurboQuant paper &middot; GPT-2 Medium</div>
  </header>

  <div class="card">
    <h2>Evaluation Results Across 3 Tasks</h2>
    <table>
      <thead>
        <tr>
          <th>Quantization Configuration</th>
          <th class="num-val">PassageQA (Accuracy %)</th>
          <th class="num-val">Summarization (ROUGE-1)</th>
          <th class="num-val">Code Completion (Accuracy %)</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>

  <div class="footer">
    Generated by standard_eval.py &middot; GPT-2 Medium &middot; Seed: 42
  </div>
</div>
</body>
</html>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
