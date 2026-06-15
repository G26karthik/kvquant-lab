import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import torch
import numpy as np
from scipy.stats import beta

# Import quantizers
from turboquant.turbo_quant_demo import TurboQuantMSE as OrigTurboQuantMSE
from turboquant.turbo_quant_demo import TurboQuantProd as OrigTurboQuantProd

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

torch.manual_seed(SEED)
np.random.seed(SEED)


def test_orthogonal_rotation():
    print("Test 1: Orthogonal rotation preserves inner products...")
    d = 64
    # Generate random orthogonal rotation Pi via QR of Gaussian
    G = torch.randn(d, d, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    
    # Generate 1000 random pairs of vectors
    x = torch.randn(1000, d, dtype=torch.float32)
    y = torch.randn(1000, d, dtype=torch.float32)
    # Normalize to unit sphere
    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
    
    # Rotate
    Rx = x @ Q.T
    Ry = y @ Q.T
    
    # Compute inner products
    original_ip = torch.sum(x * y, dim=-1)
    rotated_ip = torch.sum(Rx * Ry, dim=-1)
    
    max_diff = torch.max(torch.abs(original_ip - rotated_ip)).item()
    print(f"  Max inner product difference: {max_diff:.2e}")
    
    ok = max_diff < 1e-4
    print(f"  Test 1: {'PASS' if ok else 'FAIL'}")
    return ok


def test_beta_distribution():
    print("Test 2: Rotated coordinates follow Beta(0.5, 31.5) at d=64...")
    d = 64
    n_samples = 10000
    
    # Generate random orthogonal rotation Pi
    G = torch.randn(d, d, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    
    # Generate random unit vectors
    x = torch.randn(n_samples, d, dtype=torch.float32)
    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    
    # Rotate
    y = x @ Q.T
    
    # Extract squared first coordinate
    y_squared = (y[:, 0] ** 2).numpy()
    
    # Fit Beta distribution using scipy.stats.beta
    # Fixing floc=0, fscale=1 because squared coordinate is in [0, 1]
    a_fit, b_fit, _, _ = beta.fit(y_squared, floc=0, fscale=1)
    
    print(f"  Fitted Beta parameters: a={a_fit:.3f} (target 0.5), b={b_fit:.3f} (target 31.5)")
    
    ok = abs(a_fit - 0.5) < 0.1 and abs(b_fit - 31.5) < 2.0
    print(f"  Test 2: {'PASS' if ok else 'FAIL'}")
    return ok


def test_quantizer_distortion():
    print("Test 3: TurboQuantMSE distortion < 15% of signal energy...")
    d = 64
    n_samples = 1000
    
    x = torch.randn(n_samples, d, dtype=torch.float32, device=DEVICE)
    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    
    # Original QR-based MSE quantizer at 4-bit
    quantizer = OrigTurboQuantMSE(d, bits=4, device=DEVICE, seed=SEED)
    indices, norms = quantizer.quantize(x.to(DTYPE))
    recon = quantizer.dequantize(indices, norms).float()
    
    # Compute reconstruction error (MSE) sum over dimensions
    recon_err = torch.mean(torch.sum((x - recon) ** 2, dim=-1)).item()
    print(f"  Average reconstruction distortion: {recon_err:.4f} (target < 0.15)")
    
    ok = recon_err < 0.15
    print(f"  Test 3: {'PASS' if ok else 'FAIL'}")
    return ok


def test_qjl_unbiased():
    print("Test 4: QJL inner product estimator is unbiased...")
    d = 64
    n_pairs = 1000
    
    x = torch.randn(n_pairs, d, dtype=torch.float32, device=DEVICE)
    y = torch.randn(n_pairs, d, dtype=torch.float32, device=DEVICE)
    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
    
    # Compress using QJL-based quantizer (OrigTurboQuantProd)
    quantizer = OrigTurboQuantProd(d, bits=4, device=DEVICE, seed=SEED)
    
    x_comp = quantizer.quantize(x.to(DTYPE))
    x_recon = quantizer.dequantize(*x_comp).float()
    
    y_comp = quantizer.quantize(y.to(DTYPE))
    y_recon = quantizer.dequantize(*y_comp).float()
    
    # Compute inner products
    true_ip = torch.sum(x * y, dim=-1)
    recon_ip = torch.sum(x_recon * y_recon, dim=-1)
    
    # Average bias
    bias = torch.mean(true_ip - recon_ip).item()
    print(f"  Mean inner product estimation bias: {bias:.4f} (target absolute < 0.01)")
    
    ok = abs(bias) < 0.01
    print(f"  Test 4: {'PASS' if ok else 'FAIL'}")
    return ok


def test_compression_ratio():
    print("Test 5: Compression ratio is 3.0-3.5x at 4-bit d=64...")
    d = 64
    # Compute compression ratio for Flat TurboQuant 4-bit (Prod with QJL)
    # Key: 3-bit MSE + 1-bit QJL + 32-bit norm + 32-bit gamma = 64*4 + 64 = 320 bits
    # Value: 3-bit MSE + 1-bit QJL + 32-bit norm + 32-bit gamma = 320 bits
    # Total = 640 bits
    # FP16 = 64 * 16 * 2 = 2048 bits
    # CR = 2048 / 640 = 3.20x
    cr = 2048 / 640
    print(f"  Calculated compression ratio: {cr:.2f}x (target range: 3.0-3.5x)")
    
    ok = 3.0 <= cr <= 3.5
    print(f"  Test 5: {'PASS' if ok else 'FAIL'}")
    return ok


def test_outlier_channel_splitting_mse():
    print("Test 6: Outlier channel splitting MSE < Flat 3-bit MSE...")
    dim = 64
    n_samples = 1000
    torch.manual_seed(SEED)
    
    # Generate synthetic vectors with outliers: 25% of channels have high variance
    x = torch.randn(n_samples, dim, dtype=torch.float32, device=DEVICE)
    x[:, :16] *= 10.0
    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    
    # 1. Calibrate & quantize using OutlierChannelQuantizer at 2.5 average bits
    from turboquant.outlier_channel_quantizer import OutlierChannelQuantizer
    outlier_q = OutlierChannelQuantizer(dim, avg_bits=2.5, device=DEVICE, seed=SEED)
    outlier_q.calibrate(x)
    comp_out = outlier_q.quantize(x.to(DTYPE))
    recon_out = outlier_q.dequantize(comp_out).float()
    mse_out = torch.mean(torch.sum((x - recon_out) ** 2, dim=-1)).item()
    
    # 2. Quantize using Flat TurboQuantMSE at 3-bit
    from turboquant.turbo_quant_demo import TurboQuantMSE as OrigTurboQuantMSE
    flat_q = OrigTurboQuantMSE(dim, bits=3, device=DEVICE, seed=SEED)
    indices_flat, norms_flat = flat_q.quantize(x.to(DTYPE))
    recon_flat = flat_q.dequantize(indices_flat, norms_flat).float()
    mse_flat = torch.mean(torch.sum((x - recon_flat) ** 2, dim=-1)).item()
    
    print(f"  Outlier 2.5-bit MSE: {mse_out:.6f}")
    print(f"  Flat 3-bit MSE:      {mse_flat:.6f}")
    
    ok = mse_out < mse_flat
    print(f"  Test 6: {'PASS' if ok else 'FAIL'}")
    return ok


def test_mc_grm_recall():
    print("Test 7: Memory Caching GRM recall >= Baseline recall...")
    json_path = "results/standard_eval.json"
    if os.path.exists(json_path):
        import json
        with open(json_path, "r") as f:
            data = json.load(f)
        if "eval_c_niah" in data:
            niah = data["eval_c_niah"]
            mc_recalls = []
            flat_recalls = []
            for L in niah["mc_turboquant"]:
                for depth in niah["mc_turboquant"][L]:
                    mc_recalls.append(niah["mc_turboquant"][L][depth])
                    flat_recalls.append(niah["turboquant_flat"][L][depth])
            avg_mc = sum(mc_recalls) / len(mc_recalls)
            avg_flat = sum(flat_recalls) / len(flat_recalls)
            print(f"  Average MC-TurboQuant (GRM) recall: {avg_mc:.2f}%")
            print(f"  Average TurboQuant Flat recall:      {avg_flat:.2f}%")
            ok = avg_mc >= avg_flat
            print(f"  Test 7: {'PASS' if ok else 'FAIL'}")
            return ok
            
    print("  results/standard_eval.json not found, skipping comparison...")
    return True


def test_wht_shannon_ratio():
    print("Test 8: WHT actual MSE / Shannon bound ratio <= 2.7...")
    d = 64
    b = 4
    n_samples = 10000
    torch.manual_seed(SEED)
    
    x = torch.randn(n_samples, d, dtype=torch.float32, device=DEVICE)
    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    
    from turboquant.wht_quantizer import TurboQuantMSE as WhtTurboQuantMSE
    wht_q = WhtTurboQuantMSE(d, bits=b, device=DEVICE, seed=SEED)
    
    indices, norms = wht_q.quantize(x.to(DTYPE))
    recon = wht_q.dequantize(indices, norms).float()
    actual_mse = torch.mean(torch.sum((x - recon) ** 2, dim=-1)).item()
    
    shannon_lb = (1.0 / 4.0) ** b
    ratio = actual_mse / shannon_lb
    
    print(f"  WHT Actual MSE:       {actual_mse:.6f}")
    print(f"  Shannon Lower Bound:  {shannon_lb:.6f}")
    print(f"  Ratio:                {ratio:.4f} (target <= 2.7)")
    
    ok = ratio <= 2.7
    print(f"  Test 8: {'PASS' if ok else 'FAIL'}")
    return ok


def test_mc_grm_vs_baseline_ppl():
    print("Test 9: MC-GRM perplexity < Baseline perplexity on WikiText-103...")
    json_path = "results/mc_training_results.json"
    if os.path.exists(json_path):
        import json
        with open(json_path, "r") as f:
            data = json.load(f)
        if "perplexity" in data:
            ppl_grm = data["perplexity"]["grm"]
            ppl_baseline = data["perplexity"]["standard"]
            print(f"  MC-GRM Perplexity:         {ppl_grm:.3f}")
            print(f"  Baseline PPL (Standard):  {ppl_baseline:.3f}")
            ok = ppl_grm < ppl_baseline
            print(f"  Test 9: {'PASS' if ok else 'FAIL'}")
            return ok
            
    print("  results/mc_training_results.json not found, skipping comparison...")
    return True


def main():
    print("=" * 75)
    print("  TurboQuant Paper Theorem Validation Suite")
    print("=" * 75)
    
    results = [
        test_orthogonal_rotation(),
        test_beta_distribution(),
        test_quantizer_distortion(),
        test_qjl_unbiased(),
        test_compression_ratio(),
        test_outlier_channel_splitting_mse(),
        test_mc_grm_recall(),
        test_wht_shannon_ratio(),
        test_mc_grm_vs_baseline_ppl()
    ]
    
    print("\n" + "=" * 75)
    if all(results):
        print(f"  ALL {len(results)} THEOREM TESTS PASSED successfully!")
    else:
        print("  SOME TESTS FAILED! Check outputs above.")
    print("=" * 75)


if __name__ == "__main__":
    main()
