import os
import sys
import json
import math
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast, GradScaler
from transformers import GPT2Tokenizer
from datasets import load_dataset

# Ensure project root is on path so benchmarks package is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------
# Gated & Residual Causal Linear Attention Module (Parallel Training)
# ----------------------------------------------------------------------
class MCCausalLinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, segment_size=64, variant="standard"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.segment_size = segment_size
        self.variant = variant  # "standard", "residual", "grm"
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        if variant == "grm":
            self.u_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        H = self.n_heads
        d_h = self.head_dim
        
        q = self.q_proj(x).view(B, L, H, d_h).transpose(1, 2)  # [B, H, L, d_h]
        k = self.k_proj(x).view(B, L, H, d_h).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, d_h).transpose(1, 2)
        
        if self.variant == "grm" and L % self.segment_size == 0:
            S = self.segment_size
            n_segments = L // S
            u = self.u_proj(x).view(B, L, H, d_h).transpose(1, 2)
            x_seg = x.view(B, n_segments, S, D)
            mean_x_seg = x_seg.mean(dim=-2)
            mean_u_seg = self.u_proj(mean_x_seg).view(B, n_segments, H, d_h).transpose(1, 2)
        
        # Force all linear attention math to float32 by disabling autocast
        with autocast('cuda', enabled=False):
            q = q.float()
            k = k.float()
            v = v.float()
            
            # Feature map phi(x) = elu(x) + 1
            q_phi = torch.nn.functional.elu(q) + 1.0
            k_phi = torch.nn.functional.elu(k) + 1.0
            
            if self.variant == "standard" or L % self.segment_size != 0:
                scores = torch.matmul(q_phi, k_phi.transpose(-1, -2))  # [B, H, L, L]
                mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
                scores = scores * mask

                denom = scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                num = torch.matmul(scores, v)
                out = num / denom
                
            else:
                S = self.segment_size
                n_segments = L // S
                
                # Reshape into segments
                q_phi_seg = q_phi.view(B, H, n_segments, S, d_h)
                k_phi_seg = k_phi.view(B, H, n_segments, S, d_h)
                v_seg = v.view(B, H, n_segments, S, d_h)
                
                # 1. Compute intra-segment causal outputs (online memory)
                scores_online = torch.matmul(q_phi_seg, k_phi_seg.transpose(-1, -2))
                mask_online = torch.tril(torch.ones(S, S, device=x.device)).view(1, 1, 1, S, S)
                scores_online = scores_online * mask_online
                
                num_online = torch.matmul(scores_online, v_seg)  # [B, H, n_segments, S, d_h]
                denom_online = scores_online.sum(dim=-1, keepdim=True) + 1e-8
                
                # 2. Compute segment memory representations
                M_seg = torch.matmul(k_phi_seg.transpose(-1, -2), v_seg)
                Z_seg = k_phi_seg.sum(dim=-2, keepdim=True).transpose(-1, -2)
                
                # Pad at the start to represent shift (past segments only)
                M_seg_padded = torch.cat([torch.zeros(B, H, 1, d_h, d_h, device=x.device), M_seg], dim=2)
                Z_seg_padded = torch.cat([torch.zeros(B, H, 1, d_h, 1, device=x.device), Z_seg], dim=2)
                
                if self.variant == "residual":
                    # Cumulative sum of past memories
                    M_sum = torch.cumsum(M_seg_padded, dim=2)[:, :, :-1]
                    Z_sum = torch.cumsum(Z_seg_padded, dim=2)[:, :, :-1]
                    
                    num_past = torch.matmul(q_phi_seg, M_sum)
                    denom_past = torch.matmul(q_phi_seg, Z_sum) + 1e-8
                    
                    # Equation 7 summation
                    out = (num_online / denom_online) + (num_past / denom_past)
                    
                elif self.variant == "grm":
                    u_32 = u.float()
                    mean_u_seg_32 = mean_u_seg.float()
                    u_seg = u_32.view(B, H, n_segments, S, d_h)
                    
                    # Compute query-to-segment similarities
                    scores_gate = torch.matmul(u_seg, mean_u_seg_32.unsqueeze(2).transpose(-1, -2))  # [B, H, n_segments, S, n_segments]
                    
                    # Causal gating mask: cannot attend to future segments (i > s)
                    mask_gate = torch.tril(torch.ones(n_segments, n_segments, device=x.device)).view(1, 1, n_segments, 1, n_segments)
                    scores_gate = scores_gate.masked_fill(mask_gate == 0, float('-inf'))
                    
                    gamma = torch.softmax(scores_gate, dim=-1)  # [B, H, n_segments, S, n_segments]
                    
                    # Broadcast and compute output of each segment for each query position
                    q_exp = q_phi_seg.unsqueeze(3)  # [B, H, n_segments, 1, S, d_h]
                    M_exp = M_seg.unsqueeze(2)      # [B, H, 1, n_segments, d_h, d_h]
                    Z_exp = Z_seg.unsqueeze(2)      # [B, H, 1, n_segments, d_h, 1]
                    
                    num_all = torch.matmul(q_exp, M_exp)
                    denom_all = torch.matmul(q_exp, Z_exp) + 1e-8
                    out_all = num_all / denom_all  # [B, H, n_segments, n_segments, S, d_h]
                    
                    # For i == s (online segment), use online causal output
                    diag_mask = torch.eye(n_segments, device=x.device).view(1, 1, n_segments, n_segments, 1, 1)
                    online_exp = (num_online / denom_online).unsqueeze(3)
                    
                    segment_outputs = out_all * (1 - diag_mask) + online_exp * diag_mask
                    
                    # Multiply by gated weights
                    gamma_perm = gamma.transpose(3, 4).unsqueeze(-1)
                    out = (segment_outputs * gamma_perm).sum(dim=3)
                    
            out = out.to(x.dtype)
            
        out = out.view(B, H, L, d_h).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)

# ----------------------------------------------------------------------
# Linear Attention Language Model (Transformer Decoder wrapper)
# ----------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, segment_size=64, variant="standard"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MCCausalLinearAttention(d_model, n_heads, segment_size, variant)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class LinearAttentionLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6, max_seq_len=512, segment_size=64, variant="standard"):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, segment_size, variant)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying
        
        nn.init.normal_(self.pos_emb, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, targets=None):
        B, L = input_ids.shape
        x = self.token_emb(input_ids) + self.pos_emb[:, :L, :]
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            
        return logits, loss

# ----------------------------------------------------------------------
# Data Streaming Generator
# ----------------------------------------------------------------------
class TokenizedDatasetIter:
    def __init__(self, dataset, tokenizer, batch_size=8, seq_len=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.iter = iter(dataset)
        self.token_buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.token_buffer) < self.batch_size * self.seq_len:
            try:
                example = next(self.iter)
                text = example["text"]
                if not text.strip():
                    continue
                ids = self.tokenizer.encode(text)
                self.token_buffer.extend(ids)
            except StopIteration:
                if len(self.token_buffer) >= self.seq_len:
                    break
                raise StopIteration
                
        if len(self.token_buffer) < self.seq_len:
            raise StopIteration
            
        # Slice out batch
        n_tokens = (len(self.token_buffer) // self.seq_len) * self.seq_len
        n_tokens = min(n_tokens, self.batch_size * self.seq_len)
        
        tokens = self.token_buffer[:n_tokens]
        self.token_buffer = self.token_buffer[n_tokens:]
        
        # Reshape to [B, seq_len]
        batch_ids = torch.tensor(tokens).view(-1, self.seq_len)
        
        # Input and target (shifted by 1)
        inputs = batch_ids[:, :-1]
        targets = batch_ids[:, 1:]
        return inputs, targets

# ----------------------------------------------------------------------
# Train & Eval Pipeline
# ----------------------------------------------------------------------
def train_model(variant, steps, batch_size, lr, tokenizer):
    print(f"\nTraining {variant.upper()} model for {steps} steps...")
    
    WARMUP_STEPS = 500  # Linear LR warmup to avoid early explosion

    # WikiText-103 Streaming
    ds_train = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    train_iter = TokenizedDatasetIter(ds_train, tokenizer, batch_size=batch_size, seq_len=513)
    
    model = LinearAttentionLM(vocab_size=len(tokenizer), variant=variant).to(DEVICE)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Linear warmup then cosine decay
    def lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            return float(current_step + 1) / float(max(1, WARMUP_STEPS))
        progress = float(current_step - WARMUP_STEPS) / float(max(1, steps - WARMUP_STEPS))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')
    
    losses = []
    nan_skips = 0
    t0 = time.time()
    
    step = 0
    while step < steps:
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = TokenizedDatasetIter(ds_train, tokenizer, batch_size=batch_size, seq_len=513)
            inputs, targets = next(train_iter)
            
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            _, loss = model(inputs, targets)

        # --- NaN/Inf guard: skip bad batches instead of poisoning weights ---
        if not torch.isfinite(loss):
            nan_skips += 1
            scaler.update()  # keep scaler state consistent
            scheduler.step()
            step += 1
            if step % 100 == 0 or step == steps:
                elapsed = time.time() - t0
                print(f"  Step {step}/{steps} | Loss: NaN/Inf (skipped, total skips={nan_skips}) | Time: {elapsed:.1f}s")
            continue
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        losses.append(loss.item())
        step += 1
        
        if step % 100 == 0 or step == steps:
            elapsed = time.time() - t0
            print(f"  Step {step}/{steps} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
            
    return model, losses

def evaluate_perplexity(model, tokenizer, stride=256, context=512):
    model.eval()
    print("Evaluating perplexity on wikitext-103 test split...")
    ds_test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(ds_test["text"])
    encodings = tokenizer(text, return_tensors="pt")
    
    seq_len = encodings.input_ids.size(1)
    max_eval_tokens = min(seq_len, 20000)  # Evaluate on first 20k tokens for speed
    
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, max_eval_tokens, stride):
        end_loc = min(begin_loc + context, max_eval_tokens)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            with autocast('cuda'):
                logits, _ = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss.float() * trg_len)
                
        prev_end_loc = end_loc
        if end_loc == max_eval_tokens:
            break
            
    perplexity = math.exp(torch.stack(nlls).sum().item() / max_eval_tokens)
    return perplexity

def run_synthetic_niah(model, tokenizer, n_trials=5, L=512):
    model.eval()
    from benchmarks.standard_eval import generate_niah_prompt_with_depth
    successes = 0
    for trial in range(n_trials):
        torch.manual_seed(SEED + trial)
        np.random.seed(SEED + trial)
        secret_code = f"{np.random.randint(1000, 9999)}"
        
        prompt_ids = generate_niah_prompt_with_depth(tokenizer, L - 4, 0.5, secret_code)
        input_ids = torch.tensor([prompt_ids], device=DEVICE)
        
        with torch.no_grad():
            with autocast('cuda'):
                logits, _ = model(input_ids)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = [next_token.item()]
                
                # Generate 4 tokens
                for step in range(3):
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    logits, _ = model(input_ids)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated.append(next_token.item())
                    
        gen_text = tokenizer.decode(generated).strip()
        if secret_code in gen_text:
            successes += 1
            
    return (successes / n_trials) * 100.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    
    if args.fast:
        args.steps = 500
        
    print("=" * 80)
    print("Running RNN Training Experiment (WikiText-103 LM + Linear Attention)")
    print(f"Device: {DEVICE} | Steps: {args.steps} | Batch Size: {args.batch_size}")
    print("=" * 80)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    variants = ["standard", "residual", "grm"]
    trained_models = {}
    losses_dict = {}
    ppl_results = {}
    niah_results = {}
    
    for var in variants:
        model, losses = train_model(var, args.steps, args.batch_size, args.lr, tokenizer)
        trained_models[var] = model
        losses_dict[var] = losses
        
        # Save checkpoint
        checkpoint_dir = "results/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"lm_{var}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved {var} model checkpoint to {checkpoint_path}")
        
        # Evaluate
        ppl = evaluate_perplexity(model, tokenizer)
        ppl_results[var] = ppl
        print(f"  {var.upper()} Test Perplexity: {ppl:.3f}")
        
        # Evaluate Synthetic NIAH
        niah_acc = run_synthetic_niah(model, tokenizer)
        niah_results[var] = niah_acc
        print(f"  {var.upper()} NIAH Retrieval Accuracy: {niah_acc:.1f}%")
        
    # Save training results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    out_json = {
        "steps": args.steps,
        "perplexity": ppl_results,
        "niah_accuracy": niah_results,
        "losses": {var: losses_dict[var][::10] for var in losses_dict}  # Downsample loss curve for plotting
    }
    
    json_path = os.path.join(results_dir, "mc_training_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved training results to {json_path}")
    
    # Generate HTML report
    html_path = os.path.join(results_dir, "mc_training_results.html")
    generate_html(out_json, html_path)
    print(f"Saved HTML report to {html_path}")

def generate_html(data, output_path):
    ppl = data["perplexity"]
    niah = data["niah_accuracy"]
    steps = data["steps"]
    
    rows = ""
    for var in ["standard", "residual", "grm"]:
        label = {
            "standard": "Baseline Linear Attention",
            "residual": "MC-Residual Linear Attention",
            "grm": "MC-GRM Linear Attention"
        }[var]
        
        rows += f"""
        <tr>
            <td class="dim-val">{label}</td>
            <td class="num-val highlight">{ppl[var]:.3f}</td>
            <td class="num-val">{(niah[var]):.1f}%</td>
        </tr>
        """
        
    # Generate SVG line chart for losses
    loss_svg = ""
    losses = data["losses"]
    
    # Simple SVG line drawing helper
    svg_width = 600
    svg_height = 300
    padding = 40
    
    max_loss = max(max(losses[var]) for var in losses)
    min_loss = min(min(losses[var]) for var in losses)
    
    def get_x(idx, total):
        return padding + (idx / (total - 1)) * (svg_width - 2 * padding)
        
    def get_y(val):
        return padding + (1.0 - (val - min_loss) / (max_loss - min_loss)) * (svg_height - 2 * padding)
        
    colors = {
        "standard": "#ff5555",
        "residual": "#8be9fd",
        "grm": "#50fa7b"
    }
    
    svg_paths = ""
    for var in ["standard", "residual", "grm"]:
        y_vals = losses[var]
        pts = []
        for i, val in enumerate(y_vals):
            x_pos = get_x(i, len(y_vals))
            y_pos = get_y(val)
            pts.append(f"{x_pos:.1f},{y_pos:.1f}")
        path_data = "M" + " L".join(pts)
        svg_paths += f'<path d="{path_data}" fill="none" stroke="{colors[var]}" stroke-width="2" />\n'
        
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Memory Caching Training Experiment Results</title>
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
  .chart-container {{
    display: flex;
    justify-content: center;
    background: #12141d;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 20px;
    margin-top: 20px;
  }}
  .legend {{
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 15px;
    font-size: 12px;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .color-box {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
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
    <h1>Linear Attention Memory Caching Training Results</h1>
    <div class="subtitle">Comparing Standard, MC-Residual, and MC-GRM Linear Attention models trained on WikiText-103</div>
  </header>

  <div class="card">
    <h2>Language Modeling and In-Context Recall Evaluation (d_model=256, 6 layers)</h2>
    <table>
      <thead>
        <tr>
          <th>Attention Variant</th>
          <th class="num-val">WikiText-103 Test Perplexity</th>
          <th class="num-val">NIAH Retrieval Accuracy (%)</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Training Loss Convergence ({steps} Steps)</h2>
    <div class="chart-container">
      <svg width="{svg_width}" height="{svg_height}">
        <!-- Grid lines -->
        <line x1="{padding}" y1="{padding}" x2="{svg_width - padding}" y2="{padding}" stroke="#2a2d3a" />
        <line x1="{padding}" y1="{svg_height - padding}" x2="{svg_width - padding}" y2="{svg_height - padding}" stroke="#2a2d3a" />
        <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{svg_height - padding}" stroke="#2a2d3a" />
        <line x1="{svg_width - padding}" y1="{padding}" x2="{svg_width - padding}" y2="{svg_height - padding}" stroke="#2a2d3a" />
        
        {svg_paths}
      </svg>
    </div>
    <div class="legend">
      <div class="legend-item">
        <div class="color-box" style="background:#ff5555;"></div>
        <span>Baseline Linear Attention</span>
      </div>
      <div class="legend-item">
        <div class="color-box" style="background:#8be9fd;"></div>
        <span>MC-Residual</span>
      </div>
      <div class="legend-item">
        <div class="color-box" style="background:#50fa7b;"></div>
        <span>MC-GRM</span>
      </div>
    </div>
  </div>

  <div class="footer">
    Generated by train_mc_rnn.py &middot; WikiText-103 &middot; Training steps: {steps}
  </div>
</div>
</body>
</html>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()
