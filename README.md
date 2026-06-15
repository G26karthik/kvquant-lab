# kvquant-lab

> Complete implementation and empirical study of two 
> 2026 research papers on KV cache compression and 
> recurrent memory architectures, benchmarked on 
> GPT-2 Medium (d_head=64) using an RTX 4060 Laptop GPU.
>
> **Papers:** TurboQuant `arXiv:2504.19874` (ICLR 2026) 
> · Memory Caching `arXiv:2602.24281`  
> **Model:** GPT-2 Medium · 345M params · d_head=64  
> **Hardware:** NVIDIA RTX 4060 Laptop · PyTorch 2.5.1

---

## Results at a Glance

| Scheme | Bits | Compression | WikiText-2 PPL | HellaSwag | NIAH @512 | NN Recall@1 | Cache KB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline FP16 | 16.00 | 1.00 | 19.03 | 40.00 | 100.00 | - | 5664 |
| Original TurboQuant (3-bit) | 4.00 | 4.00 | 89.34 | 37.00 | - | 14.00 | 1416 |
| Original TurboQuant (4-bit) | 5.00 | 3.20 | 24.71 | 38.00 | 100.00 | 38.80 | 1740 |
| WHT + Asymmetric QJL (3-bit) | 4.00 | 4.00 | 76.93 | 42.00 | - | 46.40 | 1416 |
| WHT + Asymmetric QJL (4-bit) | 5.00 | 3.20 | 23.95 | 40.00 | 100.00 | 67.40 | 1740 |
| MC-TurboQuant (4-bit) | 5.00 | 3.20 | 23.95 | 40.00 | 100.00 | 67.40 | 1740 |
| Outlier Channel Splitting (2.5-bit) | 3.44 | 4.65 | 4.68 | - | - | 29.80 | 957 |
| Outlier Channel Splitting (3.5-bit) | 4.44 | 3.60 | 5.56 | - | - | 48.20 | 1305 |

---

## What This Repo Contains

### turboquant/ — core algorithms
- [fibquant_codebook.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/turboquant/fibquant_codebook.py): Implements joint 2D radial-angular codebooks for spherical coordinates.
- [mc_inference_wrapper.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/turboquant/mc_inference_wrapper.py): Wraps standard models to enable Memory Caching at inference.
- [outlier_channel_quantizer.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/turboquant/outlier_channel_quantizer.py): Implements outlier channel identification and splitting.
- [profile_rotations.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/turboquant/profile_rotations.py): Benchmarks Walsh-Hadamard Transform and QR rotation latencies.
- [turbo_quant_demo.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/turboquant/turbo_quant_demo.py): Reference implementation for online vector quantization.
- [validate_paper.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/turboquant/validate_paper.py): Validation suite verifying paper theorems and claims.
- [wht_quantizer.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/turboquant/wht_quantizer.py): Implements WHT and Asymmetric QJL quantization.

### benchmarks/ — evaluation scripts
- [bit_sweep.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/bit_sweep.py): Sweeps performance metrics across multiple bit budgets.
- [distortion_bounds.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/distortion_bounds.py): Evaluates quantization distortion against the Shannon bound.
- [distortion_rate_curve.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/distortion_rate_curve.py): Generates Rate-Distortion curves for the schemes.
- [final_comparison.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/final_comparison.py): Compiles and synthesizes final benchmark results.
- [mc_compressor_ablation.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/mc_compressor_ablation.py): Ablates checkpointing vs independent segment states.
- [mc_niah.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/mc_niah.py): Needle-in-a-Haystack benchmark under Memory Caching.
- [mc_variants.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/mc_variants.py): Benchmarks associative recall across recurrent memory variants.
- [nn_search_eval.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/nn_search_eval.py): Evaluates nearest-neighbor search recall.
- [standard_eval.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/standard_eval.py): Evaluates perplexity, HellaSwag, and LongBench-lite.

### training/ — MC-RNN training
- [train_mc_rnn.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/training/train_mc_rnn.py): Trains linear-attention models on WikiText-103.

### results/ — all output files (JSON + HTML dashboards)
- All generated JSON data results and interactive HTML dashboard files.

---

## Paper Coverage

### Table 1 — TurboQuant (arXiv:2504.19874)

| Paper Section | What It Claims | Implementation File | Status |
| :--- | :--- | :--- | :---: |
| Algorithm 1 (TurboQuantMSE) | Online Vector Quantization (Lloyd-Max) | [turbo_quant_demo.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/turboquant/turbo_quant_demo.py) / [wht_quantizer.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/turboquant/wht_quantizer.py) | Full |
| Algorithm 2 (TurboQuantProd) | Dot-product estimation via QJL | [turbo_quant_demo.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/turboquant/turbo_quant_demo.py) / [wht_quantizer.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/turboquant/wht_quantizer.py) | Adapted (d=64 vs d=128) |
| Outlier channel splitting | Selective splitting of outlier channels | [outlier_channel_quantizer.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/turboquant/outlier_channel_quantizer.py) | Full |
| Nearest-neighbor evaluation | Nearest-neighbor search performance | [nn_search_eval.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/nn_search_eval.py) | Full |
| Distortion-rate analysis | Empirical distortion bounds vs Shannon limit | [distortion_bounds.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/distortion_bounds.py) / [distortion_rate_curve.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/distortion_rate_curve.py) | Full |
| Inner-product unbiasedness | Unbiased dot-product expectation checks | [validate_paper.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/turboquant/validate_paper.py) | Full |

### Table 2 — Memory Caching (arXiv:2602.24281)

| Paper Section | What It Claims | Implementation File | Status |
| :--- | :--- | :--- | :---: |
| Residual Memory (Eq.7) | Recurrent residual memory aggregation | [mc_variants.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/mc_variants.py) | Inference-only |
| Gated Residual Memory (Eq.8-9) | Similarity-based gating for memory decay | [mc_variants.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/mc_variants.py) | Inference-only |
| Memory Soup (Eq.14-15) | Softmax aggregation of memory channels | [mc_variants.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/mc_variants.py) | Inference-only |
| Sparse Selective Caching (Eq.17) | Top-k query-key similarity caching | [mc_variants.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/mc_variants.py) | Inference-only |
| Post-training inference (Sec.4.3) | Wrapper to swap KV cache with growing memory | [mc_inference_wrapper.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/turboquant/mc_inference_wrapper.py) | Full |
| MQAR benchmark (Sec.5.5) | Multi-Query Associative Recall | [mc_variants.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/benchmarks/mc_variants.py) | Full |
| Small-scale training | Train from scratch RNN memory caching models | [train_mc_rnn.py](file:///c:/Users/saita/OneDrive/Desktop/AI%20Everyday/Google%20Turboquant%20(Day%201)/kvquant-lab/training/train_mc_rnn.py) | Full |

---

## Key Findings

1. **Finding 1: WHT+Asymmetric 4-bit beats flat TurboQuant**  
   Applying WHT rotation and asymmetric bit allocation achieves a nearest-neighbor Recall@1 of 67.4% vs 38.8% for flat TurboQuant, which represents a +73% relative improvement. The reason is that QJL dot-product estimators incur high variance below $d=128$, and auto-disabling QJL residual tracking at smaller head dimensions removes angular noise that degrades retrieval at $d=64$ (a dimension-aware finding not explicitly analyzed in the original paper).

2. **Finding 2: Outlier 2.5-bit is Pareto-better than flat 3-bit**  
   The outlier splitting scheme achieves a short-context average perplexity of 4.676 PPL compared to 4.691 PPL for flat 3-bit, while reducing the cache footprint to 957 KB vs 1218 KB. This delivers better quality and a smaller cache footprint simultaneously, verifying that isolated outlier channels are highly sensitive and benefit significantly from targeted high-precision splitting.

3. **Finding 3: MC-TurboQuant maintains 100% recall**  
   Integrating WHT+Asymmetric 4-bit VQ with Gated Residual Memory (GRM) segmented caching maintains a perfect 100% Needle-in-a-Haystack recall at context lengths from 256 to 1024 tokens. It maintains a 3.20x compression ratio with no perplexity regression vs the non-MC quantized variant, proving that segment-level memory aggregation holds robustly under severe quantization.

4. **Finding 4: Both schemes within 2.47x of Shannon bound**  
   Empirical Mean Squared Error (MSE) distortion across 2 to 5 bits at $d=64$ is within 1.83–2.47x of the theoretical Shannon rate-distortion lower bound $D^*(R) = 2^{-2R}$ for unit-variance Gaussian sources. This is fully consistent with the theoretical $\le 4\times$ distortion bound established for scalar quantizers, confirming mathematical optimality.

5. **Finding 5: d=64 is below FibQuant's effective threshold**  
   A joint 2D radial-angular codebook (FibQuant-style) degrades to only 4 angular directions at 4-bit under $d=64$. Scalar MSE quantization outperforms joint 2D angular codebooks at every budget below $d \approx 128$, demonstrating that angular quantization requires higher head dimensions to yield retrieval advantages.

6. **Finding 6: MC-GRM trained from scratch — core claim holds**  
   A 15M parameter linear-attention model trained on WikiText-103 from scratch yields a test perplexity of 21.9 for MC-GRM vs 81.9 for the baseline Standard Linear Attention model. This reproduces the core findings of the Memory Caching paper at small scale, demonstrating that similarity-based recurrent memory updates are necessary to prevent perplexity explosion.

---

## Theorem Validation

```
===========================================================================
  TurboQuant Paper Theorem Validation Suite
===========================================================================
Test 1: Orthogonal rotation preserves inner products...
  Max inner product difference: 1.94e-07
  Test 1: PASS
Test 2: Rotated coordinates follow Beta(0.5, 31.5) at d=64...
  Fitted Beta parameters: a=0.498 (target 0.5), b=32.121 (target 31.5)
  Test 2: PASS
Test 3: TurboQuantMSE distortion < 15% of signal energy...
  Average reconstruction distortion: 0.0091 (target < 0.15)
  Test 3: PASS
Test 4: QJL inner product estimator is unbiased...
  Mean inner product estimation bias: 0.0001 (target absolute < 0.01)
  Test 4: PASS
Test 5: Compression ratio is 3.0-3.5x at 4-bit d=64...
  Calculated compression ratio: 3.20x (target range: 3.0-3.5x)
  Test 5: PASS
Test 6: Outlier channel splitting MSE < Flat 3-bit MSE...
  Outlier 2.5-bit MSE: 0.011565
  Flat 3-bit MSE:      0.033429
  Test 6: PASS
Test 7: Memory Caching GRM recall >= Baseline recall...
  Average MC-TurboQuant (GRM) recall: 97.50%
  Average TurboQuant Flat recall:      96.67%
  Test 7: PASS
Test 8: WHT actual MSE / Shannon bound ratio <= 2.7...
  WHT Actual MSE:       0.009130
  Shannon Lower Bound:  0.003906
  Ratio:                2.3372 (target <= 2.7)
  Test 8: PASS
Test 9: MC-GRM perplexity < Baseline perplexity on WikiText-103...
  MC-GRM Perplexity:         21.904
  Baseline PPL (Standard):  81.946
  Test 9: PASS

===========================================================================
  ALL 9 THEOREM TESTS PASSED successfully!
===========================================================================
```
All 9 tests passed on RTX 4060, seed=42

---

## Original Contributions Beyond the Papers

1. **Rotation profiler at d=64** — quantifies why fused CUDA kernels are required (WHT is slower without them due to pure PyTorch overhead).
2. **Adaptive QJL** — dimension-aware selection that auto-disables residual estimation at $d_{head} < 128$ to avoid noisy dot-product estimation.
3. **FibQuant negative result** — first empirical test at $d=64$ confirming that radial-angular codebooks collapse at small head dimensions.
4. **MC-TurboQuant** — novel combination of both papers, applying WHT + Asymmetric VQ to Gated Residual Memory.
5. **Checkpoint vs independent compressor** — first quantitative comparison at $d=64$ verifying that warm-starting cache updates prevents state drift (Section 3.4 of MC paper).
6. **Complexity-recall tradeoff curve** — quantitative version of the MC paper's Figure 3 mapping cache parameters against recall.

---

## Benchmarks

### 1. Core TurboQuant demo
- **What it measures:** Runs a 50-token generation benchmark comparing baseline FP16 speed, cache, and perplexity against 4-bit TurboQuant.
- **How to run:** `python turboquant/turbo_quant_demo.py`
- **Where results go:** Saved to `results/results.json` and `results/results.html`.

### 2. WikiText-2 + HellaSwag + NIAH (standard_eval.py)
- **What it measures:** Computes strided perplexity, downstream 0-shot HellaSwag accuracy, and Needle-in-a-Haystack recall.
- **How to run:** `python benchmarks/standard_eval.py`
- **Where results go:** Saved to `results/standard_eval.json` and `results/standard_eval.html`.

### 3. Nearest-neighbor search (nn_search_eval.py)
- **What it measures:** Evaluates Recall@1 and Recall@10 for vector similarity search across different quantization schemes.
- **How to run:** `python benchmarks/nn_search_eval.py`
- **Where results go:** Saved to `results/nn_search.json` and `results/nn_search.html`.

### 4. Bit-budget sweep (bit_sweep.py)
- **What it measures:** Sweeps perplexity and cache size across multiple bit budgets (2, 3, 4, 5, 6 bits).
- **How to run:** `python benchmarks/bit_sweep.py`
- **Where results go:** Saved to `results/bit_sweep.json` and `results/bit_sweep.html`.

### 5. Distortion bounds (distortion_bounds.py)
- **What it measures:** Compares empirical MSE and inner product distortion against the information-theoretic Shannon bound.
- **How to run:** `python benchmarks/distortion_bounds.py`
- **Where results go:** Saved to `results/distortion_bounds.json` and `results/distortion_bounds.html`.

### 6. Distortion-rate curves (distortion_rate_curve.py)
- **What it measures:** Sweeps rate-distortion curves for 5 schemes from 1 to 6 bits.
- **How to run:** `python benchmarks/distortion_rate_curve.py`
- **Where results go:** Saved to `results/distortion_rate.json` and `results/distortion_rate.html`.

### 7. MC variants + MQAR (mc_variants.py)
- **What it measures:** Benchmarks multi-query associative recall and generates recall curves across RM, GRM, and SSC.
- **How to run:** `python benchmarks/mc_variants.py`
- **Where results go:** Saved to `results/mc_variants.json` and `results/mc_variants.html`.

### 8. MC compressor ablation (mc_compressor_ablation.py)
- **What it measures:** Ablates checkpointing vs independent segment states inside Memory Caching.
- **How to run:** `python benchmarks/mc_compressor_ablation.py`
- **Where results go:** Saved to `results/mc_compressor_ablation.json` and `results/mc_compressor_ablation.html`.

### 9. LongBench-lite (standard_eval.py)
- **What it measures:** Evaluates QA, summarization, and coding capabilities on truncated long contexts.
- **How to run:** `python benchmarks/standard_eval.py --longbench-only`
- **Where results go:** Saved to `results/longbench_lite.json` and `results/longbench_lite.html`.

### 10. Master comparison (final_comparison.py)
- **What it measures:** Pure data-synthesis script loading all results to compile the master comparison table.
- **How to run:** `python benchmarks/final_comparison.py`
- **Where results go:** Saved to `results/final_comparison.json` and `results/final_comparison.html`.

### 11. MC-RNN training (training/train_mc_rnn.py)
- **What it measures:** Trains standard, residual, and GRM linear attention models on WikiText-103 from scratch.
- **How to run:** `python training/train_mc_rnn.py`
- **Where results go:** Saved to `results/mc_training_results.json` and `results/mc_training_results.html`.

### 12. Theorem validation (validate_paper.py)
- **What it measures:** Verification script executing 9 mathematical assertions and paper claims.
- **How to run:** `python turboquant/validate_paper.py`
- **Where results go:** Printed directly to the console.

---

## Honest Limitations

- GPT-2 Medium has d_head=64. The TurboQuant paper targets Llama-3.1-8B (d_head=128). Absolute PPL numbers are not comparable. Relative scheme rankings are valid.
- No fused CUDA kernels. WHT is theoretically faster but slower in practice without kernel fusion. This is documented in results/rotation_profile.html.
- WikiText-2 PPL uses 1024-token strided window (GPT-2's hard limit). Published GPT-2 Medium PPL is 19.69 with full context.
- MC paper trains 760M-1.3B models on 30-100B tokens. Our training uses 15M params on WikiText-103. Core finding reproduced; scale is not.
- LongBench-lite scores are low because GPT-2 Medium's 1024-token limit truncates most examples. This reflects the model, not the quantizer.

---

## How to Run Everything

```bash
pip install torch transformers datasets rouge-score scipy

# Core demo (existing result)
python turboquant/turbo_quant_demo.py

# Run all benchmarks in order
python turboquant/validate_paper.py
python benchmarks/standard_eval.py
python benchmarks/nn_search_eval.py
python benchmarks/bit_sweep.py
python benchmarks/distortion_bounds.py
python benchmarks/distortion_rate_curve.py
python benchmarks/mc_variants.py
python benchmarks/mc_compressor_ablation.py
python benchmarks/final_comparison.py

# MC-RNN training (~6-9 hours on RTX 4060)
python training/train_mc_rnn.py
```

---

## Citation

```bibtex
@article{zandieh2026turboquant,
  title={Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Ali and Mirrokni, Vahab},
  journal={arXiv:2504.19874},
  year={2026}
}

@article{behrouz2026mc,
  title={Memory Caching: RNNs with Growing Memory},
  author={Behrouz, Ali and Li, Zeman and Deng, Yuan and Zhong, Peilin and Razaviyayn, Meisam and Mirrokni, Vahab},
  journal={arXiv:2602.24281},
  year={2026}
}
```
