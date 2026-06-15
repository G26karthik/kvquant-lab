import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc
import json
import math
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import standard quantized cache for compression if quantize_cache is True
from turboquant.wht_quantizer import TurboQuantKVCache

# Constants
SEED = 42
MODEL_NAME = "gpt2-medium"
MAX_NEW_TOKENS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ----------------------------------------------------------------------
# Segmented Cache State per Layer for Inference
# ----------------------------------------------------------------------
class LayerMCInferenceCache:
    def __init__(self, layer_idx: int, d_head: int, device: str, segment_size: int,
                 quantize_cache: bool, quantizer):
        self.layer_idx = layer_idx
        self.d_head = d_head
        self.device = device
        self.segment_size = segment_size
        self.quantize_cache = quantize_cache
        self.quantizer = quantizer  # Pair of (key_quant, val_quant)
        
        # Completed segments
        self.completed = []
        
        # Current online segment
        self.curr_k = None
        self.curr_v = None

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
            
            if self.quantize_cache and self.quantizer is not None:
                k_q = self.quantizer[0].quantize(k_seg)
                v_q = self.quantizer[1].quantize(v_seg)
                self.completed.append((k_q, v_q))
            else:
                self.completed.append((k_seg, v_seg))
                
            self.curr_k = self.curr_k[:, :, self.segment_size:, :]
            self.curr_v = self.curr_v[:, :, self.segment_size:, :]

    def retrieve_segment(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.completed[idx]
        if self.quantize_cache and self.quantizer is not None:
            k_q, v_q = item
            k_hat = self.quantizer[0].dequantize(*k_q)
            v_hat = self.quantizer[1].dequantize(*v_q)
            return k_hat, v_hat
        else:
            return item

    def get_footprint_bits(self) -> int:
        total_bits = 0
        for item in self.completed:
            if self.quantize_cache and self.quantizer is not None:
                k_q, v_q = item
                k_indices, k_norms = k_q[0], k_q[1]
                # Key size
                total_bits += k_indices.numel() * self.quantizer[0].bits
                total_bits += k_norms.numel() * 32
                # Value size
                v_indices, v_norms = v_q
                total_bits += v_indices.numel() * self.quantizer[1].bits
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
# Active Wrapper Global Reference and Hook Functions
# ----------------------------------------------------------------------
ACTIVE_WRAPPER: Optional['MCInferenceWrapper'] = None

def mc_inference_attention_forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, **kwargs):
    if ACTIVE_WRAPPER is None:
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
    layer_cache = ACTIVE_WRAPPER.caches[self.layer_idx]
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
    variant = ACTIVE_WRAPPER.variant
    
    if variant == "rm":
        # Residual Memory summation
        attn_output = torch.zeros_like(q_32)
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
            attn_output += torch.matmul(probs, v_seg.float())
            k_start += k_len
            
    elif variant == "grm":
        # Gated Residual Memory (LSE-gated)
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
            
    elif variant == "ssc":
        # Sparse Selective Caching
        outputs_list = []
        r_list_full = []
        past_segments = segments[:-1]
        curr_segment = segments[-1]
        
        past_r_list = []
        for k_seg, _ in past_segments:
            mean_k = torch.mean(k_seg.float(), dim=2)
            score = torch.sum(q_32 * mean_k.unsqueeze(2), dim=-1)
            past_r_list.append(score)
            
        if len(past_r_list) > 0:
            past_r = torch.stack(past_r_list, dim=-1)
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
            j_indices = torch.arange(k_len, device=query_states.device).unsqueeze(0) + (i * ACTIVE_WRAPPER.segment_size)
            mask = j_indices > t_indices
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            all_masked = mask.all(dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            safe_scores = scores.masked_fill(all_masked, 0.0)
            probs = torch.softmax(safe_scores, dim=-1).masked_fill(all_masked, 0.0)
            outputs_list.append(torch.matmul(probs, v_seg.float()))
            
            if i < len(segments) - 1:
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

    # Downcast and project back
    attn_output = attn_output.to(query_states.dtype)
    attn_output = attn_output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)
    
    return attn_output, None


def patch_inference_attention(model):
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    if not hasattr(GPT2Attention, "_orig_forward"):
        GPT2Attention._orig_forward = GPT2Attention.forward
    GPT2Attention.forward = mc_inference_attention_forward
    for i, block in enumerate(model.transformer.h):
        block.attn.layer_idx = i


def unpatch_inference_attention():
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    if hasattr(GPT2Attention, "_orig_forward"):
        GPT2Attention.forward = GPT2Attention._orig_forward


# ----------------------------------------------------------------------
# MCInferenceWrapper Class
# ----------------------------------------------------------------------
class MCInferenceWrapper:
    """
    Memory Caching post-training inference wrapper (Section 4.3).
    Wraps standard GPT-2 models for segmented caching during generation.
    """
    def __init__(self, model, segment_size=64, variant='grm',
                 quantize_cache=True, quantizer=None):
        self.model = model
        self.segment_size = segment_size
        self.variant = variant
        self.quantize_cache = quantize_cache
        
        self.num_layers = model.config.n_layer
        self.d_head = model.config.n_embd // model.config.n_head
        self.device = next(model.parameters()).device
        
        if quantize_cache:
            if quantizer is None:
                self.quantizer = TurboQuantKVCache(self.num_layers, self.d_head, bits=4, device=str(self.device))
            else:
                self.quantizer = quantizer
        else:
            self.quantizer = None
            
        self.caches = {}
        for li in range(self.num_layers):
            layer_quant = None
            if self.quantizer is not None:
                layer_quant = (self.quantizer.key_quantizers[li], self.quantizer.val_quantizers[li])
            self.caches[li] = LayerMCInferenceCache(
                layer_idx=li, d_head=self.d_head, device=str(self.device),
                segment_size=segment_size, quantize_cache=quantize_cache,
                quantizer=layer_quant
            )

    def cache_footprint_kb(self) -> float:
        total_bits = sum(c.get_footprint_bits() for c in self.caches.values())
        return total_bits / (8 * 1024)

    def generate(self, input_ids: torch.Tensor, max_new_tokens=50) -> torch.Tensor:
        global ACTIVE_WRAPPER
        ACTIVE_WRAPPER = self
        
        # Clear caches
        for li in self.caches:
            self.caches[li].clear()
            
        patch_inference_attention(self.model)
        
        generated_ids = input_ids.clone()
        next_token = input_ids
        
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=False)
            logits = outputs.logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            for step in range(max_new_tokens - 1):
                pos = torch.tensor([[input_ids.shape[1] + step]], device=self.device)
                outputs = self.model(next_token, position_ids=pos, use_cache=False)
                logits = outputs.logits
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                if next_token.item() == 50256:  # EOS token
                    break
                    
        unpatch_inference_attention()
        ACTIVE_WRAPPER = None
        
        return generated_ids


# ----------------------------------------------------------------------
# Demo and Perplexity Helpers
# ----------------------------------------------------------------------
def compute_perplexity(model, tokenizer, text: str) -> float:
    encodings = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(encodings.input_ids, labels=encodings.input_ids)
    return math.exp(outputs.loss.item())


def main():
    print("=" * 75)
    print("  MC Post-Training Inference Wrapper Demo")
    print("=" * 75)
    
    print("Loading model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    prompts = [
        "The future of artificial intelligence in healthcare will",
        "Quantum computing represents a fundamental shift in",
        "The most significant challenge facing climate science today is",
        "In the field of natural language processing, transformer models have",
    ]
    
    variants = ["rm", "grm", "ssc"]
    results = {}
    
    for var in variants:
        print(f"\nRunning wrapper for variant: {var} (quantize_cache=True)...")
        wrapper = MCInferenceWrapper(model, segment_size=64, variant=var, quantize_cache=True)
        
        var_ppls = []
        var_speeds = []
        var_sizes = []
        
        for i, prompt in enumerate(prompts):
            print(f"  [Prompt {i+1}/4] {prompt[:40]}...")
            t0 = time.perf_counter()
            gen_ids = wrapper.generate(tokenizer.encode(prompt, return_tensors="pt").to(DEVICE), max_new_tokens=MAX_NEW_TOKENS)
            t1 = time.perf_counter()
            
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            ppl = compute_perplexity(model, tokenizer, gen_text)
            
            elapsed = t1 - t0
            n_tokens = gen_ids.shape[1] - len(tokenizer.encode(prompt))
            speed = n_tokens / elapsed
            size_kb = wrapper.cache_footprint_kb()
            
            var_ppls.append(ppl)
            var_speeds.append(speed)
            var_sizes.append(size_kb)
            print(f"    Speed: {speed:.2f} tok/s | PPL: {ppl:.3f} | Footprint: {size_kb:.2f} KB")
            
        results[var] = {
            "avg_perplexity": float(np.mean(var_ppls)),
            "avg_speed_tok_sec": float(np.mean(var_speeds)),
            "avg_cache_size_kb": float(np.mean(var_sizes))
        }
        
    print("\n[Averages]")
    for var in results:
        print(f"  {var:4s} -> PPL: {results[var]['avg_perplexity']:.4f} | Speed: {results[var]['avg_speed_tok_sec']:.2f} tok/s | Footprint: {results[var]['avg_cache_size_kb']:.2f} KB")
        
    # Save to JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), "results")
    
    out_json = {
        "results": results
    }
    
    json_path = os.path.join(results_dir, "mc_inference.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nJSON results saved to {json_path}")
    
    # Generate HTML
    html_path = os.path.join(results_dir, "mc_inference.html")
    generate_html(results, html_path)
    print(f"HTML report saved to {html_path}")


def generate_html(results: Dict, output_path: str):
    rows = ""
    var_labels = {
        "rm": "Residual Memory (summation)",
        "grm": "Gated Residual Memory (LSE-gated)",
        "ssc": "Sparse Selective Caching (MoE top-2)"
    }
    
    for var in ["rm", "grm", "ssc"]:
        rows += f"""
        <tr>
            <td class="dim-val">{var_labels[var]}</td>
            <td class="num-val highlight">{results[var]['avg_perplexity']:.4f}</td>
            <td class="num-val">{results[var]['avg_speed_tok_sec']:.2f} tok/s</td>
            <td class="num-val">{results[var]['avg_cache_size_kb']:.2f} KB</td>
        </tr>
        """
        
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MC Post-Training Inference Dashboard</title>
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
    <h1>Post-Training Memory Caching Inference Wrapper Evaluation</h1>
    <div class="subtitle">Comparing RM, GRM, and SSC with WHT + Asymmetric 4-bit Quantized Segment Cache &middot; GPT-2 Medium</div>
  </header>

  <div class="card">
    <h2>Average Generation Perplexity, Speed, and Cache Footprint</h2>
    <table>
      <thead>
        <tr>
          <th>Aggregation Scheme</th>
          <th class="num-val">Avg Perplexity</th>
          <th class="num-val">Generation Speed</th>
          <th class="num-val">Avg Cache Size</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>

  <div class="footer">
    Generated by mc_inference_wrapper.py &middot; GPT-2 Medium &middot; Segment Size: 64 &middot; Seed: 42
  </div>
</div>
</body>
</html>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
