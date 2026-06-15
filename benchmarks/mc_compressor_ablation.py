import os
import sys
import gc
import json
import math
import time
import random
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Global variables for the patched attention layers
CURRENT_MC_CACHES = {}
USE_MC_ATTENTION = False

# ----------------------------------------------------------------------
# Segmented Cache State per Layer
# ----------------------------------------------------------------------
class AblationLayerMCCache:
    def __init__(self, layer_idx: int, d_head: int, bits: int, device: str, strategy: str, compressed: bool = True, segment_size: int = 64):
        self.layer_idx = layer_idx
        self.d_head = d_head
        self.bits = bits
        self.device = device
        self.strategy = strategy  # "checkpoint" or "independent"
        self.compressed = compressed
        self.segment_size = segment_size
        
        self.completed = []  # List of (k_seg, v_seg) or (k_quant, v_quant)
        self.curr_k = None
        self.curr_v = None
        
        if compressed:
            from turboquant.wht_quantizer import TurboQuantMSE, TurboQuantProd
            self.key_quantizer = TurboQuantProd(d_head, bits, device, seed=SEED + layer_idx * 100)
            self.val_quantizer = TurboQuantMSE(d_head, bits, device, seed=SEED + layer_idx * 100 + 50)

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor):
        if self.curr_k is None:
            if self.strategy == "checkpoint" and len(self.completed) > 0:
                last_k, last_v = self.retrieve_segment(len(self.completed) - 1)
                self.curr_k = torch.cat([last_k, new_k], dim=2)
                self.curr_v = torch.cat([last_v, new_v], dim=2)
            else:
                self.curr_k = new_k
                self.curr_v = new_v
        else:
            self.curr_k = torch.cat([self.curr_k, new_k], dim=2)
            self.curr_v = torch.cat([self.curr_v, new_v], dim=2)
            
        if self.strategy == "independent":
            while self.curr_k.shape[2] >= self.segment_size:
                k_seg = self.curr_k[:, :, :self.segment_size, :]
                v_seg = self.curr_v[:, :, :self.segment_size, :]
                
                if self.compressed:
                    self.completed.append((self.key_quantizer.quantize(k_seg), self.val_quantizer.quantize(v_seg)))
                else:
                    self.completed.append((k_seg, v_seg))
                    
                self.curr_k = self.curr_k[:, :, self.segment_size:, :]
                self.curr_v = self.curr_v[:, :, self.segment_size:, :]
        else:
            # Checkpoint strategy
            target_len = (len(self.completed) + 1) * self.segment_size
            while self.curr_k.shape[2] >= target_len:
                k_seg = self.curr_k[:, :, :target_len, :]
                v_seg = self.curr_v[:, :, :target_len, :]
                
                if self.compressed:
                    self.completed.append((self.key_quantizer.quantize(k_seg), self.val_quantizer.quantize(v_seg)))
                else:
                    self.completed.append((k_seg, v_seg))
                    
                # Keep remainder
                self.curr_k = self.curr_k[:, :, target_len:, :]
                self.curr_v = self.curr_v[:, :, target_len:, :]
                
                # Prepend the last completed segment
                last_k, last_v = self.retrieve_segment(len(self.completed) - 1)
                self.curr_k = torch.cat([last_k, self.curr_k], dim=2)
                self.curr_v = torch.cat([last_v, self.curr_v], dim=2)
                
                target_len = (len(self.completed) + 1) * self.segment_size

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
# Attention Patch Forward
# ----------------------------------------------------------------------
def ablation_attention_forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, **kwargs):
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

    # Retrieve layer cache
    layer_cache = CURRENT_MC_CACHES[self.layer_idx]
    layer_cache.update(key_states, value_states)

    # Reconstruct all active segments
    segments = []
    for idx in range(len(layer_cache.completed)):
        segments.append(layer_cache.retrieve_segment(idx))

    if layer_cache.curr_k is not None and layer_cache.curr_k.shape[2] > 0:
        segments.append((layer_cache.curr_k, layer_cache.curr_v))

    outputs_list = []
    r_list = []
    head_dim = query_states.size(-1)
    q_len = query_states.shape[2]

    # Calculate total sequence length and starting query index
    total_seq_len = sum(k_seg.shape[2] for k_seg, _ in segments)
    q_start = total_seq_len - q_len

    # Upcast query states to float32 for high precision attention calculation
    q_32 = query_states.float()

    k_start = 0
    for k_seg, v_seg in segments:
        k_len = k_seg.shape[2]
        
        # Upcast segment key/value to float32
        k_seg_32 = k_seg.float()
        v_seg_32 = v_seg.float()
        
        # Segment attention in float32
        scores = torch.matmul(q_32, k_seg_32.transpose(-1, -2)) / math.sqrt(head_dim)
        
        # Apply causal mask: k_start is 0 for checkpoint strategy (as segments are cumulative)
        seg_k_start = 0 if layer_cache.strategy == "checkpoint" else k_start
        
        t_indices = torch.arange(q_len, device=query_states.device).unsqueeze(1) + q_start
        j_indices = torch.arange(k_len, device=query_states.device).unsqueeze(0) + seg_k_start
        mask = j_indices > t_indices  # Shape (q_len, k_len)
        mask_4d = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask_4d, float('-inf'))

        # Safe softmax
        all_masked = mask.all(dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        safe_scores = scores.masked_fill(all_masked, 0.0)
        probs = torch.softmax(safe_scores, dim=-1)
        probs = probs.masked_fill(all_masked, 0.0)

        output_i = torch.matmul(probs, v_seg_32)
        outputs_list.append(output_i)

        # Relevance gating score: LogSumExp pooling
        r_i = torch.logsumexp(scores, dim=-1)
        r_list.append(r_i)

        k_start += k_len

    # Compute gating softmax across segments in float32
    r = torch.stack(r_list, dim=-1)  # (batch, heads, q_len, n_segments)
    gamma = torch.softmax(r, dim=-1)  # (batch, heads, q_len, n_segments)

    # Sum gated segment outputs in float32
    attn_output = torch.zeros_like(q_32)
    for i, output_i in enumerate(outputs_list):
        attn_output += gamma[..., i].unsqueeze(-1) * output_i

    # Downcast back to float16
    attn_output = attn_output.to(query_states.dtype)

    # Projection back
    attn_output = attn_output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    return attn_output, None

# ----------------------------------------------------------------------
# Monkey-patching Helpers
# ----------------------------------------------------------------------
def patch_attention(model):
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    if not hasattr(GPT2Attention, "_orig_forward"):
        GPT2Attention._orig_forward = GPT2Attention.forward
    GPT2Attention.forward = ablation_attention_forward
    for i, block in enumerate(model.transformer.h):
        block.attn.layer_idx = i

def unpatch_attention():
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    if hasattr(GPT2Attention, "_orig_forward"):
        GPT2Attention.forward = GPT2Attention._orig_forward

# ----------------------------------------------------------------------
# Synthetic Prompt Generator with depth control (reused from mc_niah.py)
# ----------------------------------------------------------------------
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

    split_idx = int(depth * len(dist_ids))
    dist_pre = dist_ids[:split_idx]
    dist_post = dist_ids[split_idx:]

    full_ids = preamble_ids + dist_pre + fact_ids + dist_post + question_ids
    assert len(full_ids) == L
    return full_ids

# ----------------------------------------------------------------------
# Generation & Perplexity Helpers
# ----------------------------------------------------------------------
def compute_perplexity(model, tokenizer, text: str) -> float:
    encodings = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(encodings.input_ids, labels=encodings.input_ids)
    return math.exp(outputs.loss.item())

def run_autoregressive_generation(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> Tuple[torch.Tensor, float, float]:
    # Custom generation loop that uses position_ids correctly with patched attention
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=DEVICE)
    generated_ids = input_ids.clone()
    next_token = input_ids
    
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        
        for step in range(max_new_tokens - 1):
            pos = torch.tensor([[input_ids.shape[1] + step]], device=DEVICE)
            outputs = model(next_token, position_ids=pos, use_cache=False)
            logits = outputs.logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if next_token.item() == 50256:  # EOS
                break
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    n_tokens = generated_ids.shape[1] - input_ids.shape[1]
    speed = n_tokens / elapsed
    return generated_ids, speed, elapsed

# ----------------------------------------------------------------------
# Main ablation benchmark
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("Running Checkpoint vs Independent Memory Caching Compressor Ablation...")
    print("=" * 80)
    
    print("Loading model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    num_layers = model.config.n_layer
    d_head = model.config.n_embd // model.config.n_head
    
    strategies = ["checkpoint", "independent"]
    results = {strat: {} for strat in strategies}
    
    # ------------------------------------------------------------------
    # Part 1: Standard Prompt Benchmarking (PPL, speed, footprint)
    # ------------------------------------------------------------------
    prompts = [
        "The future of artificial intelligence in healthcare will",
        "Quantum computing represents a fundamental shift in",
        "The most significant challenge facing climate science today is",
        "In the field of natural language processing, transformer models have",
    ]
    
    global CURRENT_MC_CACHES, USE_MC_ATTENTION
    patch_attention(model)
    
    for strat in strategies:
        print(f"\nPart 1: Evaluating standard prompts for strategy: {strat}...")
        
        # Configure caches
        CURRENT_MC_CACHES = {}
        for li in range(num_layers):
            CURRENT_MC_CACHES[li] = AblationLayerMCCache(
                layer_idx=li, d_head=d_head, bits=4, device=DEVICE,
                strategy=strat, compressed=True, segment_size=64
            )
        USE_MC_ATTENTION = True
        
        var_ppls = []
        var_speeds = []
        var_sizes = []
        
        for i, prompt in enumerate(prompts):
            # Clear caches
            for li in CURRENT_MC_CACHES:
                CURRENT_MC_CACHES[li].clear()
                
            gen_ids, speed, elapsed = run_autoregressive_generation(model, tokenizer, prompt, max_new_tokens=50)
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            ppl = compute_perplexity(model, tokenizer, gen_text)
            
            # Cache footprint in KB
            total_bits = sum(c.get_footprint_bits() for c in CURRENT_MC_CACHES.values())
            size_kb = total_bits / (8 * 1024)
            
            var_ppls.append(ppl)
            var_speeds.append(speed)
            var_sizes.append(size_kb)
            print(f"  [Prompt {i+1}/4] Speed: {speed:.2f} tok/s | PPL: {ppl:.3f} | Footprint: {size_kb:.2f} KB")
            
        results[strat]["avg_perplexity"] = float(np.mean(var_ppls))
        results[strat]["avg_speed_tok_sec"] = float(np.mean(var_speeds))
        results[strat]["avg_cache_size_kb"] = float(np.mean(var_sizes))
        
    # ------------------------------------------------------------------
    # Part 2: Needle-in-a-Haystack (NIAH) Task
    # ------------------------------------------------------------------
    lengths = [256, 512, 768, 1024]
    depths = {"Early": 0.1, "Middle": 0.5, "Late": 0.9}
    n_trials = 5
    
    for strat in strategies:
        print(f"\nPart 2: Evaluating NIAH for strategy: {strat}...")
        
        # Configure caches
        CURRENT_MC_CACHES = {}
        for li in range(num_layers):
            CURRENT_MC_CACHES[li] = AblationLayerMCCache(
                layer_idx=li, d_head=d_head, bits=4, device=DEVICE,
                strategy=strat, compressed=True, segment_size=64
            )
        USE_MC_ATTENTION = True
        
        matrix = {l: {d: 0.0 for d in depths} for l in lengths}
        
        for L in lengths:
            for d_name, d_val in depths.items():
                successes = 0
                for trial in range(n_trials):
                    random.seed(42 + L + int(d_val * 100) + trial)
                    secret_code = f"{random.randint(1000, 9999)}"
                    
                    prompt_ids = generate_niah_prompt_with_depth(tokenizer, L - 4, d_val, secret_code)
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
                        
                matrix[L][d_name] = (successes / n_trials) * 100.0
                print(f"  L={L}, Depth={d_name} -> {matrix[L][d_name]:.0f}%")
                
        results[strat]["niah"] = matrix
        
    unpatch_attention()
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "mc_compressor_ablation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON results to {json_path}")
    
    html_path = os.path.join(results_dir, "mc_compressor_ablation.html")
    generate_html(results, html_path)
    print(f"Saved HTML report to {html_path}")

def generate_html(data, output_path):
    rows = ""
    for strat in ["checkpoint", "independent"]:
        label = "Checkpoint (Stateful/Warm-start)" if strat == "checkpoint" else "Independent (Stateless/Fresh)"
        niah = {str(k): v for k, v in data[strat]["niah"].items()}
        r_L256 = niah["256"]
        r_L512 = niah["512"]
        r_L768 = niah["768"]
        r_L1024 = niah["1024"]
        
        rows += f"""
        <tr>
            <td class="dim-val" rowspan="4" style="vertical-align: middle; border-bottom: 2px solid #2a2d3a;">{label}</td>
            <td class="num-val">256</td>
            <td class="num-val">{r_L256['Early']:.0f}%</td>
            <td class="num-val">{r_L256['Middle']:.0f}%</td>
            <td class="num-val" style="border-right: 1px solid #2a2d3a;">{r_L256['Late']:.0f}%</td>
            <td class="num-val highlight" rowspan="4" style="vertical-align: middle; border-bottom: 2px solid #2a2d3a;">{data[strat]['avg_perplexity']:.4f}</td>
            <td class="num-val" rowspan="4" style="vertical-align: middle; border-bottom: 2px solid #2a2d3a;">{data[strat]['avg_speed_tok_sec']:.2f}</td>
            <td class="num-val" rowspan="4" style="vertical-align: middle; border-bottom: 2px solid #2a2d3a;">{data[strat]['avg_cache_size_kb']:.2f} KB</td>
        </tr>
        <tr>
            <td class="num-val">512</td>
            <td class="num-val">{r_L512['Early']:.0f}%</td>
            <td class="num-val">{r_L512['Middle']:.0f}%</td>
            <td class="num-val" style="border-right: 1px solid #2a2d3a;">{r_L512['Late']:.0f}%</td>
        </tr>
        <tr>
            <td class="num-val">768</td>
            <td class="num-val">{r_L768['Early']:.0f}%</td>
            <td class="num-val">{r_L768['Middle']:.0f}%</td>
            <td class="num-val" style="border-right: 1px solid #2a2d3a;">{r_L768['Late']:.0f}%</td>
        </tr>
        <tr style="border-bottom: 2px solid #2a2d3a;">
            <td class="num-val">1024</td>
            <td class="num-val">{r_L1024['Early']:.0f}%</td>
            <td class="num-val">{r_L1024['Middle']:.0f}%</td>
            <td class="num-val" style="border-right: 1px solid #2a2d3a;">{r_L1024['Late']:.0f}%</td>
        </tr>
        """
        
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MC Checkpoint vs Independent Compressor Ablation</title>
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
  .commentary {{
    background: #1a1d28;
    border: 1px solid #bd93f944;
    border-left: 4px solid #bd93f9;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 30px;
    line-height: 1.6;
    font-size: 13.5px;
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
    color: #8be9fd;
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
    <h1>Memory Caching Compressor Ablation Dashboard</h1>
    <div class="subtitle">Quantitative Evaluation of Checkpoint vs Independent Memory Compressors &middot; GPT-2 Medium</div>
  </header>

  <div class="commentary">
    <strong>Section 3.4 of arXiv:2602.24281</strong> discusses this design choice explicitly but provides only qualitative guidance (<em>"each has its own advantages"</em>).
    This report provides the first quantitative comparison at d=64.
    We compare:
    <ul>
      <li><strong>Checkpoint (Stateful)</strong>: Warm-starts each segment's memory from the previous: M^(s)_0 = M^(s-1)_L.</li>
      <li><strong>Independent (Stateless)</strong>: Re-initializes memory from scratch for each segment: M^(s)_0 = 0.</li>
    </ul>
  </div>

  <div class="card">
    <h2>Performance &amp; Recall Metrics</h2>
    <table>
      <thead>
        <tr>
          <th>Strategy</th>
          <th class="num-val">L</th>
          <th class="num-val">Early</th>
          <th class="num-val">Middle</th>
          <th class="num-val" style="border-right: 1px solid #2a2d3a;">Late</th>
          <th class="num-val">Avg Perplexity</th>
          <th class="num-val">Avg Throughput</th>
          <th class="num-val">Cache Footprint</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>

  <div class="footer">
    Generated by mc_compressor_ablation.py &middot; GPT-2 Medium &middot; Segment Size: 64 &middot; Seed: 42
  </div>
</div>
</body>
</html>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()
