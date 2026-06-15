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

# Constants
SEED = 42
MODEL_NAME = "gpt2-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Global states for attention hooks
CURRENT_VARIANT = "grm"  # "rm", "grm", "ssc"
CURRENT_MC_CACHES = {}
USE_MC_ATTENTION = False


# ----------------------------------------------------------------------
# Segmented Cache State per Layer (FP16 or Compressed)
# ----------------------------------------------------------------------
class LayerMCCache:
    def __init__(self, layer_idx: int, d_head: int, device: str, segment_size: int = 32, compressed: bool = False, bits: int = 4):
        self.layer_idx = layer_idx
        self.d_head = d_head
        self.device = device
        self.segment_size = segment_size
        self.compressed = compressed
        self.bits = bits
        
        # Completed segments: each is (k_seg, v_seg) or (k_quant, v_quant)
        self.completed = []
        
        # Current online segment
        self.curr_k = None
        self.curr_v = None
        
        if compressed:
            from turboquant.wht_quantizer import TurboQuantMSE, TurboQuantProd
            self.key_quantizer = TurboQuantProd(d_head, bits, device, seed=SEED + layer_idx * 100)
            self.val_quantizer = TurboQuantMSE(d_head, bits, device, seed=SEED + layer_idx * 100 + 50)
        
    def update(self, new_k: torch.Tensor, new_v: torch.Tensor):
        if self.curr_k is None:
            self.curr_k = new_k
            self.curr_v = new_v
        else:
            self.curr_k = torch.cat([self.curr_k, new_k], dim=2)
            self.curr_v = torch.cat([self.curr_v, new_v], dim=2)
            
        # Split into completed segments
        while self.curr_k.shape[2] >= self.segment_size:
            k_seg = self.curr_k[:, :, :self.segment_size, :]
            v_seg = self.curr_v[:, :, :self.segment_size, :]
            
            if self.compressed:
                self.completed.append((self.key_quantizer.quantize(k_seg), self.val_quantizer.quantize(v_seg)))
            else:
                self.completed.append((k_seg, v_seg))
                
            self.curr_k = self.curr_k[:, :, self.segment_size:, :]
            self.curr_v = self.curr_v[:, :, self.segment_size:, :]
            
    def retrieve_segment(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.completed[idx]
        if self.compressed:
            k_hat = self.key_quantizer.dequantize(*item[0])
            v_hat = self.val_quantizer.dequantize(*item[1])
            return k_hat, v_hat
        else:
            return item

    def get_footprint_bits(self) -> int:
        total_bits = 0
        for item in self.completed:
            if self.compressed:
                k_indices, k_norms, k_qjl_signs, k_gamma = item[0]
                total_bits += k_indices.numel() * (self.bits - 1)
                total_bits += k_qjl_signs.numel() * 1
                total_bits += k_norms.numel() * 32
                total_bits += k_gamma.numel() * 32
                
                v_indices, v_norms = item[1]
                total_bits += v_indices.numel() * self.bits
                total_bits += v_norms.numel() * 32
            else:
                k_seg, v_seg = item
                total_bits += k_seg.numel() * 16 + v_seg.numel() * 16
        if self.curr_k is not None:
            total_bits += self.curr_k.numel() * 16 + self.curr_v.numel() * 16
        return total_bits

    def clear(self):
        self.completed = []
        self.curr_k = None
        self.curr_v = None


# ----------------------------------------------------------------------
# Causal Scaled Dot Product Attention Patch supporting RM, GRM, SSC
# ----------------------------------------------------------------------
def mc_attention_forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, **kwargs):
    if not USE_MC_ATTENTION:
        return self._orig_forward(hidden_states, past_key_value, cache_position, attention_mask, **kwargs)
        
    is_cross_attention = kwargs.get("encoder_hidden_states", None) is not None
    if is_cross_attention:
        return self._orig_forward(hidden_states, past_key_value, cache_position, attention_mask, **kwargs)

    # Compute Q, K, V
    query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
    shape_q = (*query_states.shape[:-1], -1, self.head_dim)
    shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

    query_states = query_states.view(shape_q).transpose(1, 2)
    key_states = key_states.view(shape_kv).transpose(1, 2)
    value_states = value_states.view(shape_kv).transpose(1, 2)

    # Retrieve and update layer cache
    layer_cache = CURRENT_MC_CACHES[self.layer_idx]
    layer_cache.update(key_states, value_states)

    # Reconstruct segments
    segments = []
    for idx in range(len(layer_cache.completed)):
        segments.append(layer_cache.retrieve_segment(idx))
    if layer_cache.curr_k is not None and layer_cache.curr_k.shape[2] > 0:
        segments.append((layer_cache.curr_k, layer_cache.curr_v))

    head_dim = query_states.size(-1)
    q_len = query_states.shape[2]
    total_seq_len = sum(k_seg.shape[2] for k_seg, _ in segments)
    q_start = total_seq_len - q_len

    q_32 = query_states.float()
    
    if CURRENT_VARIANT == "rm":
        # Variant 1 — Residual Memory: Simple summation of segment outputs (no gating)
        attn_output = torch.zeros_like(q_32)
        k_start = 0
        for k_seg, v_seg in segments:
            k_len = k_seg.shape[2]
            scores = torch.matmul(q_32, k_seg.float().transpose(-1, -2)) / math.sqrt(head_dim)
            
            # Causal mask across segments
            t_indices = torch.arange(q_len, device=query_states.device).unsqueeze(1) + q_start
            j_indices = torch.arange(k_len, device=query_states.device).unsqueeze(0) + k_start
            mask = j_indices > t_indices
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            all_masked = mask.all(dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            safe_scores = scores.masked_fill(all_masked, 0.0)
            probs = torch.softmax(safe_scores, dim=-1).masked_fill(all_masked, 0.0)
            
            attn_output += torch.matmul(probs, v_seg.float())
            k_start += k_len
            
    elif CURRENT_VARIANT == "grm":
        # Variant 2 — Gated Residual Memory: LogSumExp natural attention gating
        outputs_list = []
        r_list = []
        k_start = 0
        for k_seg, v_seg in segments:
            k_len = k_seg.shape[2]
            scores = torch.matmul(q_32, k_seg.float().transpose(-1, -2)) / math.sqrt(head_dim)
            
            t_indices = torch.arange(q_len, device=query_states.device).unsqueeze(1) + q_start
            j_indices = torch.arange(k_len, device=query_states.device).unsqueeze(0) + k_start
            mask = j_indices > t_indices
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
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
            
    elif CURRENT_VARIANT == "ssc":
        # Variant 4 — Sparse Selective Caching: Router selects top-2 past segments + current segment
        outputs_list = []
        r_list_full = []
        
        # Segments list contains completed segments + current segment
        past_segments = segments[:-1]
        curr_segment = segments[-1]
        
        # Relevance scores for past segments: r_i = <q_t, MeanPooling(K^(i))>
        past_r_list = []
        for k_seg, _ in past_segments:
            mean_k = torch.mean(k_seg.float(), dim=2)  # (batch, heads, head_dim)
            score = torch.sum(q_32 * mean_k.unsqueeze(2), dim=-1)  # (batch, heads, q_len)
            past_r_list.append(score)
            
        # Select top-K=2 indices per query position
        if len(past_r_list) > 0:
            past_r = torch.stack(past_r_list, dim=-1)  # (batch, heads, q_len, n_past)
            k_top = min(2, len(past_r_list))
            _, top_indices = torch.topk(past_r, k_top, dim=-1)
        else:
            top_indices = None
            
        mean_k_curr = torch.mean(curr_segment[0].float(), dim=2)
        r_curr = torch.sum(q_32 * mean_k_curr.unsqueeze(2), dim=-1)
        
        for i, (k_seg, v_seg) in enumerate(segments):
            k_len = k_seg.shape[2]
            scores = torch.matmul(q_32, k_seg.float().transpose(-1, -2)) / math.sqrt(head_dim)
            
            t_indices = torch.arange(q_len, device=query_states.device).unsqueeze(1) + q_start
            # approximate segment start index
            j_indices = torch.arange(k_len, device=query_states.device).unsqueeze(0) + (i * layer_cache.segment_size)
            mask = j_indices > t_indices
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            all_masked = mask.all(dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            safe_scores = scores.masked_fill(all_masked, 0.0)
            probs = torch.softmax(safe_scores, dim=-1).masked_fill(all_masked, 0.0)
            
            outputs_list.append(torch.matmul(probs, v_seg.float()))
            
            if i < len(segments) - 1:
                # Past segment: only keep relevance score if it is in top-2 selected
                score_i = past_r_list[i].clone()
                if top_indices is not None:
                    is_selected = (top_indices == i).any(dim=-1)
                    score_i[~is_selected] = float('-inf')
                r_list_full.append(score_i)
            else:
                r_list_full.append(r_curr)
                
        r = torch.stack(r_list_full, dim=-1)
        gamma = torch.softmax(r, dim=-1)
        gamma = torch.nan_to_num(gamma, nan=0.0)
        
        attn_output = torch.zeros_like(q_32)
        for i, output_i in enumerate(outputs_list):
            attn_output += gamma[..., i].unsqueeze(-1) * output_i

    attn_output = attn_output.to(query_states.dtype)
    attn_output = attn_output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)
    
    return attn_output, None


def patch_attention(model):
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    if not hasattr(GPT2Attention, "_orig_forward"):
        GPT2Attention._orig_forward = GPT2Attention.forward
    GPT2Attention.forward = mc_attention_forward
    
    for i, block in enumerate(model.transformer.h):
        block.attn.layer_idx = i


def unpatch_attention():
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    if hasattr(GPT2Attention, "_orig_forward"):
        GPT2Attention.forward = GPT2Attention._orig_forward


# ----------------------------------------------------------------------
# Variant 3 — Memory Soup Mathematical Equivalence Unit Test
# ----------------------------------------------------------------------
def test_memory_soup_equivalence():
    """
    Prove that for linear matrix-valued memory (e.g. linear attention),
    Memory Soup (Equation 14-15) is mathematically identical to GRM (Equation 9).
    """
    print("Running Memory Soup vs GRM mathematical equivalence test...")
    batch, heads, q_len, head_dim = 1, 12, 1, 64
    n_segs = 4
    seg_len = 32
    
    # Generate random queries, keys, values
    q = torch.randn(batch, heads, q_len, head_dim)
    K_list = [torch.randn(batch, heads, seg_len, head_dim) for _ in range(n_segs)]
    V_list = [torch.randn(batch, heads, seg_len, head_dim) for _ in range(n_segs)]
    
    # Gating scores (logsumexp natural attention gate)
    r_list = []
    for i in range(n_segs):
        scores = torch.matmul(q, K_list[i].transpose(-1, -2)) / math.sqrt(head_dim)
        r_list.append(torch.logsumexp(scores, dim=-1))
    r = torch.stack(r_list, dim=-1)
    gamma = torch.softmax(r, dim=-1) # (batch, heads, q_len, n_segs)
    
    # 1. GRM Linear (ensembled outputs of individual linear attention modules)
    y_grm = torch.zeros_like(q)
    for i in range(n_segs):
        g = gamma[..., i].unsqueeze(-1)
        # Linear attention segment response: q @ (K_i.T @ V_i)
        seg_attn = torch.matmul(torch.matmul(q, K_list[i].transpose(-1, -2)), V_list[i])
        y_grm += g * seg_attn
        
    # 2. Memory Soup Linear (weighted interpolation of segment memory matrices)
    # M_i = K_i.T @ V_i
    # M_soup = sum(gamma_i * M_i)
    # y_soup = q @ M_soup
    M_list = [torch.matmul(K_list[i].transpose(-1, -2), V_list[i]) for i in range(n_segs)]
    M_soup = torch.zeros(batch, heads, head_dim, head_dim)
    for i in range(n_segs):
        g = gamma[..., i].unsqueeze(-1).unsqueeze(-1) # (batch, heads, q_len, 1, 1)
        M_soup += g.squeeze(2) * M_list[i]
    y_soup = torch.matmul(q, M_soup)
    
    is_close = torch.allclose(y_grm, y_soup, atol=1e-5)
    print(f"Memory Soup vs GRM linear memory equivalence: {'PASS' if is_close else 'FAIL'}")
    return is_close


# ----------------------------------------------------------------------
# MQAR Prompt Generation
# ----------------------------------------------------------------------
def generate_mqar_prompt(tokenizer, L: int, n_pairs: int, pairs: List[Tuple[str, str]], query_key: str) -> List[int]:
    kv_decl = ", ".join([f"{k}: {v}" for k, v in pairs]) + ". "
    query_str = f"Query: {query_key} ->"
    
    kv_ids = tokenizer.encode(kv_decl)
    query_ids = tokenizer.encode(query_str)
    
    distractors = [
        "The grass is green and the sky is blue. ",
        "Many people enjoy drinking hot coffee in the morning. ",
        "A standard laptop has a keyboard and a screen. ",
        "Python is a popular programming language for data science. ",
        "The capital of France is Paris, known for the Eiffel Tower. ",
    ]
    
    dist_ids = []
    i = 0
    while len(dist_ids) + len(kv_ids) + len(query_ids) < L:
        ids = tokenizer.encode(distractors[i % len(distractors)])
        dist_ids.extend(ids)
        i += 1
        
    needed_dist_len = L - len(kv_ids) - len(query_ids)
    if needed_dist_len > 0:
        dist_ids = dist_ids[:needed_dist_len]
        full_ids = kv_ids + dist_ids + query_ids
    else:
        # If KV + Query alone exceeds target length, do not pad
        full_ids = kv_ids + query_ids
        
    return full_ids


# ----------------------------------------------------------------------
# MQAR Evaluation Loop
# ----------------------------------------------------------------------
def run_mqar_eval(model, tokenizer) -> Dict:
    lengths = [64, 128, 256]
    n_pairs_list = [4, 8, 16]
    n_trials = 10
    
    # Single-token keys and values from vocabulary
    vocab_keys = ["red", "blue", "green", "pink", "black", "white", "gold", "silver", "orange", "purple", "brown", "gray", "yellow", "cyan", "magenta", "bronze"]
    vocab_vals = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen"]
    
    variants = ["baseline", "rm", "grm", "ssc"]
    results = {v: {} for v in variants}
    
    # Pre-patch attention
    patch_attention(model)
    
    global CURRENT_VARIANT, CURRENT_MC_CACHES, USE_MC_ATTENTION
    
    for var in variants:
        print(f"\nEvaluating MQAR for variant: {var}")
        
        # Configure hooks
        if var == "baseline":
            USE_MC_ATTENTION = False
        else:
            USE_MC_ATTENTION = True
            CURRENT_VARIANT = var
            # Initialize caches
            num_layers = model.config.n_layer
            d_head = model.config.n_embd // model.config.n_head
            CURRENT_MC_CACHES = {}
            for li in range(num_layers):
                CURRENT_MC_CACHES[li] = LayerMCCache(li, d_head, DEVICE)
                
        results[var] = {}
        
        for L in lengths:
            results[var][str(L)] = {}
            for n_pairs in n_pairs_list:
                successes = 0
                for trial in range(n_trials):
                    # Set deterministic trial seed
                    random.seed(SEED + L + n_pairs + trial)
                    np.random.seed(SEED + L + n_pairs + trial)
                    
                    # Sample pairs
                    indices = np.random.choice(len(vocab_keys), n_pairs, replace=False)
                    pairs = [(vocab_keys[idx], vocab_vals[idx]) for idx in indices]
                    
                    # Pick a query key from the pairs
                    q_key, q_val = random.choice(pairs)
                    
                    prompt_ids = generate_mqar_prompt(tokenizer, L, n_pairs, pairs, q_key)
                    input_ids = torch.tensor([prompt_ids], device=DEVICE)
                    
                    # Clear caches
                    if var != "baseline":
                        for li in CURRENT_MC_CACHES:
                            CURRENT_MC_CACHES[li].clear()
                            
                    with torch.no_grad():
                        outputs = model(input_ids, use_cache=False)
                        logits = outputs.logits
                        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True).item()
                        
                    # Decode prediction and ground truth to check match
                    pred_str = tokenizer.decode([next_token]).strip().lower()
                    gt_str = q_val.strip().lower()
                    
                    if pred_str == gt_str:
                        successes += 1
                        
                accuracy = (successes / n_trials) * 100.0
                results[var][str(L)][str(n_pairs)] = accuracy
                print(f"  Length={L}, Pairs={n_pairs} -> Accuracy={accuracy:.1f}%")
                
    unpatch_attention()
    return results


def run_complexity_recall_curve(model, tokenizer):
    print("\n" + "=" * 70)
    print("  Running Complexity vs Recall Tradeoff Curve (Section 4.2)")
    print("=" * 70)
    
    # Segment sizes
    segment_sizes = [8, 16, 32, 64, 128, 256]
    L = 512
    n_trials = 10
    depth_val = 0.5 # Middle depth
    
    results = []
    
    # Patch attention
    patch_attention(model)
    
    global CURRENT_VARIANT, CURRENT_MC_CACHES, USE_MC_ATTENTION
    CURRENT_VARIANT = "grm"
    USE_MC_ATTENTION = True
    
    num_layers = model.config.n_layer
    d_head = model.config.n_embd // model.config.n_head
    
    from benchmarks.mc_niah import generate_niah_prompt_with_depth
    
    for S in segment_sizes:
        print(f"\nEvaluating segment size S = {S}...")
        
        N = L / S
        cost_fraction = N / L  # which is 1 / S
        
        # Initialize LayerMCCache with segment_size=S and compressed=True
        CURRENT_MC_CACHES = {}
        for li in range(num_layers):
            CURRENT_MC_CACHES[li] = LayerMCCache(
                layer_idx=li, d_head=d_head, device=DEVICE,
                segment_size=S, compressed=True, bits=4
            )
            
        successes = 0
        footprints = []
        
        for trial in range(n_trials):
            random.seed(42 + L + S + trial)
            secret_code = f"{random.randint(1000, 9999)}"
            
            prompt_ids = generate_niah_prompt_with_depth(tokenizer, L - 4, depth_val, secret_code)
            input_ids = torch.tensor([prompt_ids], device=DEVICE)
            
            # Clear caches
            for li in CURRENT_MC_CACHES:
                CURRENT_MC_CACHES[li].clear()
                
            with torch.no_grad():
                # Prefill
                outputs = model(input_ids, use_cache=False)
                logits = outputs.logits
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = [next_token.item()]
                
                # Generate 4 tokens
                for step in range(3):
                    pos = torch.tensor([[L - 4 + step]], device=DEVICE)
                    outputs = model(next_token, position_ids=pos, use_cache=False)
                    logits = outputs.logits
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated.append(next_token.item())
                    
            gen_text = tokenizer.decode(generated, skip_special_tokens=True)
            if secret_code in gen_text:
                successes += 1
                
            # Measure footprint
            total_bits = sum(c.get_footprint_bits() for c in CURRENT_MC_CACHES.values())
            footprints.append(total_bits / (8 * 1024)) # KB
            
        accuracy = (successes / n_trials) * 100.0
        avg_footprint_kb = float(np.mean(footprints))
        
        results.append({
            "segment_size": S,
            "cost_fraction": cost_fraction,
            "recall_accuracy": accuracy,
            "cache_footprint_kb": avg_footprint_kb
        })
        print(f"  Segment Size: {S} | Cost Fraction: {cost_fraction:.4f} | Recall: {accuracy:.1f}% | Footprint: {avg_footprint_kb:.2f} KB")
        
    unpatch_attention()
    USE_MC_ATTENTION = False
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "complexity_recall.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nComplexity-Recall results saved to {json_path}")
    
    html_path = os.path.join(results_dir, "complexity_recall.html")
    generate_complexity_recall_html(results, html_path)
    print(f"Complexity-Recall HTML report saved to {html_path}")


def generate_complexity_recall_html(results, output_path):
    import json
    json_data_str = json.dumps(results)
    
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MC Complexity vs Recall Tradeoff</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0f1117;
    color: #e6e6e6;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    padding: 40px 20px;
    min-height: 100vh;
  }
  .container {
    max-width: 900px;
    margin: 0 auto;
  }
  header {
    text-align: center;
    margin-bottom: 40px;
  }
  header h1 {
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
  }
  header .subtitle {
    font-size: 14px;
    color: #8b8fa3;
    font-weight: 400;
  }
  .commentary {
    background: #1a1d28;
    border: 1px solid #bd93f944;
    border-left: 4px solid #bd93f9;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 30px;
    line-height: 1.6;
    font-size: 13.5px;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
  }
  .card {
    background: #1a1d28;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 24px;
  }
  .card h2 {
    font-size: 16px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 20px;
  }
  .chart-container {
    position: relative;
    width: 100%;
    height: 400px;
    background: #12141d;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 10px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  th, td {
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid #2a2d3a;
  }
  th {
    background: #12141d;
    color: #8b8fa3;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
  }
  .num-val {
    font-variant-numeric: tabular-nums;
    text-align: right;
  }
  th.num-val {
    text-align: right;
  }
  .highlight {
    color: #50fa7b;
    font-weight: 600;
  }
  .footer {
    text-align: center;
    margin-top: 40px;
    font-size: 12px;
    color: #555;
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Complexity vs Recall Tradeoff Curve</h1>
    <div class="subtitle">Section 4.2 of arXiv:2602.24281 &middot; Gated Residual Memory at d=64</div>
  </header>

  <div class="commentary">
    Section 4.2 of the Memory Caching paper discusses how the segment size controls the tradeoff between computation and recall.
    This dashboard provides the first quantitative analysis of this tradeoff at d=64.
    The x-axis shows the fraction of Transformer cost (N/L = 1/S), and the y-axis shows the recall accuracy (%) on the NIAH task at L=512.
  </div>

  <div class="grid">
    <div class="card">
      <h2>Tradeoff Curve</h2>
      <div class="chart-container">
        <svg id="svg-curve" width="100%" height="100%" viewBox="0 0 500 350"></svg>
      </div>
    </div>
    <div class="card">
      <h2>Data Table</h2>
      <table>
        <thead>
          <tr>
            <th>Segment Size</th>
            <th class="num-val">Cost Fraction (N/L)</th>
            <th class="num-val">Recall Accuracy (%)</th>
            <th class="num-val">Cache Footprint</th>
          </tr>
        </thead>
        <tbody id="table-body">
        </tbody>
      </table>
    </div>
  </div>

  <div class="footer">
    Generated by mc_variants.py &middot; GPT-2 Medium &middot; L=512 &middot; Seed: 42
  </div>
</div>

<script>
const rawData = JSON_DATA_HOLDER;

// Populate table
const tbody = document.getElementById("table-body");
rawData.forEach(d => {
    const row = document.createElement("tr");
    row.innerHTML = `
        <td style="color:#50fa7b; font-weight:600;">S = ${d.segment_size}</td>
        <td class="num-val">${(d.cost_fraction * 100).toFixed(2)}%</td>
        <td class="num-val highlight">${d.recall_accuracy.toFixed(0)}%</td>
        <td class="num-val">${d.cache_footprint_kb.toFixed(2)} KB</td>
    `;
    tbody.appendChild(row);
});

// Draw curve
function drawCurve() {
    const svg = document.getElementById("svg-curve");
    const width = 500;
    const height = 350;
    const padding = { top: 30, right: 30, bottom: 45, left: 50 };
    
    svg.innerHTML = '';
    
    // X scale: 0 to 0.13 (13%)
    const xMin = 0.0;
    const xMax = 0.13;
    // Y scale: 0 to 100
    const yMin = 0;
    const yMax = 100;
    
    function getX(xVal) {
        return padding.left + (xVal / xMax) * (width - padding.left - padding.right);
    }
    
    function getY(yVal) {
        return padding.top + (1 - yVal / 100) * (height - padding.top - padding.bottom);
    }
    
    // Draw background grid lines (Y axis - 20% intervals)
    for (let y = 0; y <= 100; y += 20) {
        const yPos = getY(y);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", padding.left);
        line.setAttribute("y1", yPos);
        line.setAttribute("x2", width - padding.right);
        line.setAttribute("y2", yPos);
        line.setAttribute("stroke", "#2a2d3a");
        line.setAttribute("stroke-width", "1");
        svg.appendChild(line);
        
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", padding.left - 8);
        label.setAttribute("y", yPos + 4);
        label.setAttribute("fill", "#8b8fa3");
        label.setAttribute("font-size", "9");
        label.setAttribute("text-anchor", "end");
        label.textContent = y + "%";
        svg.appendChild(label);
    }
    
    // Draw background grid lines (X axis - 2% intervals)
    for (let x = 0.0; x <= xMax; x += 0.02) {
        const xPos = getX(x);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", xPos);
        line.setAttribute("y1", padding.top);
        line.setAttribute("x2", xPos);
        line.setAttribute("y2", height - padding.bottom);
        line.setAttribute("stroke", "#2a2d3a");
        line.setAttribute("stroke-width", "1");
        svg.appendChild(line);
        
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", xPos);
        label.setAttribute("y", height - padding.bottom + 15);
        label.setAttribute("fill", "#8b8fa3");
        label.setAttribute("font-size", "9");
        label.setAttribute("text-anchor", "middle");
        label.textContent = (x * 100).toFixed(0) + "%";
        svg.appendChild(label);
    }
    
    // Sort data points by cost_fraction (X) to draw a continuous line
    const sortedData = [...rawData].sort((a, b) => a.cost_fraction - b.cost_fraction);
    
    let pathD = "";
    sortedData.forEach((d, idx) => {
        const x = getX(d.cost_fraction);
        const y = getY(d.recall_accuracy);
        pathD += (idx === 0 ? "M" : "L") + x + " " + y;
    });
    
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", pathD);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "#bd93f9");
    path.setAttribute("stroke-width", "2.5");
    svg.appendChild(path);
    
    // Draw points and labels
    sortedData.forEach(d => {
        const cx = getX(d.cost_fraction);
        const cy = getY(d.recall_accuracy);
        
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", cx);
        circle.setAttribute("cy", cy);
        circle.setAttribute("r", "5");
        circle.setAttribute("fill", "#50fa7b");
        svg.appendChild(circle);
        
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", cx + 8);
        text.setAttribute("y", cy - 8);
        text.setAttribute("fill", "#ffffff");
        text.setAttribute("font-size", "10");
        text.setAttribute("font-weight", "600");
        text.textContent = `S=${d.segment_size}`;
        svg.appendChild(text);
    });
    
    // X label
    const xLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    xLabel.setAttribute("x", padding.left + (width - padding.left - padding.right)/2);
    xLabel.setAttribute("y", height - 5);
    xLabel.setAttribute("fill", "#8b8fa3");
    xLabel.setAttribute("font-size", "10");
    xLabel.setAttribute("text-anchor", "middle");
    xLabel.textContent = "Fraction of Transformer Cost (N/L = 1/S)";
    svg.appendChild(xLabel);
    
    // Y label
    const yLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    yLabel.setAttribute("transform", "rotate(-90)");
    yLabel.setAttribute("x", -(padding.top + (height - padding.top - padding.bottom)/2));
    yLabel.setAttribute("y", 15);
    yLabel.setAttribute("fill", "#8b8fa3");
    yLabel.setAttribute("font-size", "10");
    yLabel.setAttribute("text-anchor", "middle");
    yLabel.textContent = "Recall Accuracy (%)";
    svg.appendChild(yLabel);
}

drawCurve();
</script>
</body>
</html>"""
    
    final_html = html_template.replace("JSON_DATA_HOLDER", json_data_str)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)


def main():
    print("=" * 75)
    print("  Memory Caching Variants & MQAR Evaluation")
    print("=" * 75)
    
    # 1. Run Unit Test
    soup_ok = test_memory_soup_equivalence()
    if not soup_ok:
        print("ERROR: Memory Soup equivalence failed!")
        
    # 2. Run MQAR evaluation
    print("\nLoading GPT-2 Medium...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    mqar_results = run_mqar_eval(model, tokenizer)
    
    # Save to JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    out_json = {
        "memory_soup_unit_test_passed": soup_ok,
        "mqar_results": mqar_results
    }
    
    json_path = os.path.join(results_dir, "mc_variants.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nJSON results saved to {json_path}")
    
    # Generate HTML report for MQAR
    html_path = os.path.join(results_dir, "mc_variants.html")
    generate_html(mqar_results, html_path)
    print(f"HTML report saved to {html_path}")
    
    # 3. Run complexity recall tradeoff curve
    run_complexity_recall_curve(model, tokenizer)


def generate_html(results: Dict, output_path: str):
    rows = ""
    
    variants = ["baseline", "rm", "grm", "ssc"]
    var_labels = {
        "baseline": "Baseline GPT-2",
        "rm": "Residual Memory (summation)",
        "grm": "Gated Residual Memory (LSE-gated)",
        "ssc": "Sparse Selective Caching (MoE top-2)"
    }
    
    for var in variants:
        for L in ["64", "128", "256"]:
            r_L = results[var][L]
            rows += f"""
            <tr>
                <td class="dim-val">{var_labels[var]}</td>
                <td class="dim-val" style="color:#ffffff;">{L}</td>
                <td class="num-val">{(r_L['4']):.1f}%</td>
                <td class="num-val">{(r_L['8']):.1f}%</td>
                <td class="num-val highlight">{(r_L['16']):.1f}%</td>
            </tr>
            """
            
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Memory Caching Variants MQAR Dashboard</title>
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
    <h1>Memory Caching Aggregation Variants Benchmark</h1>
    <div class="subtitle">Multi-Query Associative Recall (MQAR) Task Comparison &middot; GPT-2 Medium</div>
  </header>

  <div class="card">
    <h2>MQAR Accuracy (%) by Context Length and Key-Value Pairs</h2>
    <table>
      <thead>
        <tr>
          <th>Aggregation Scheme</th>
          <th>Sequence Length (L)</th>
          <th class="num-val">4 Pairs</th>
          <th class="num-val">8 Pairs</th>
          <th class="num-val">16 Pairs</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>

  <div class="footer">
    Generated by mc_variants.py &middot; GPT-2 Medium &middot; Segment Size: 32 &middot; Seed: 42
  </div>
</div>
</body>
</html>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
