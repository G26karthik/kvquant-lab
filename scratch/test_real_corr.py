import os
import sys
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer on CPU
MODEL_NAME = "gpt2-medium"
print(f"Loading {MODEL_NAME} on CPU...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()

prompt = "The future of artificial intelligence in healthcare will"

# Hook to extract Q and K
c_attn_outputs = []
def hook_fn(module, input, output):
    c_attn_outputs.append(output)

hooks = []
for block in model.transformer.h:
    hooks.append(block.attn.c_attn.register_forward_hook(hook_fn))

# Run forward pass
input_ids = tokenizer.encode(prompt, return_tensors="pt")
with torch.no_grad():
    model(input_ids)

# Remove hooks
for h in hooks:
    h.remove()

n_embd = model.config.n_embd
n_head = model.config.n_head
d_head = n_embd // n_head

all_q = []
all_k = []
for output in c_attn_outputs:
    # output shape: (1, seq_len, 3072)
    q, k, v = output.split(n_embd, dim=-1)
    q = q.view(1, -1, n_head, d_head).transpose(1, 2)
    k = k.view(1, -1, n_head, d_head).transpose(1, 2)
    all_q.append(q)
    all_k.append(k)

all_q = torch.cat(all_q, dim=0)  # (n_layer, n_head, seq_len, d_head)
all_k = torch.cat(all_k, dim=0)  # (n_layer, n_head, seq_len, d_head)

# Normalize along the head dimension to get unit vectors
q_norm = all_q / (torch.norm(all_q, dim=-1, keepdim=True) + 1e-8)
k_norm = all_k / (torch.norm(all_k, dim=-1, keepdim=True) + 1e-8)

# Cosine similarity is the inner product of normalized vectors
cosine_sim = torch.sum(q_norm * k_norm, dim=-1)

avg_cosine = torch.mean(cosine_sim).item()
avg_abs_cosine = torch.mean(torch.abs(cosine_sim)).item()

print(f"Average Cosine Similarity (Uncentered Correlation): {avg_cosine:.6f}")
print(f"Average Absolute Cosine Similarity: {avg_abs_cosine:.6f}")
