import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from turboquant.turbo_quant_demo import TurboQuantMSE, TurboQuantProd

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
N_pairs = 10000

for d in [64, 128, 256]:
    x = torch.randn(N_pairs, d, dtype=torch.float32)
    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    y = torch.randn(N_pairs, d, dtype=torch.float32)
    y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
    true_ips = torch.sum(x * y, dim=-1)
    
    # 4-bit MSE
    mse4 = TurboQuantMSE(d, bits=4, device="cpu", seed=SEED)
    ix, nx = mse4.quantize(x)
    x_hat = mse4.dequantize(ix, nx).float()
    iy, ny = mse4.quantize(y)
    y_hat = mse4.dequantize(iy, ny).float()
    ips_mse = torch.sum(x_hat * y_hat, dim=-1)
    
    # 4-bit QJL
    qjl4 = TurboQuantProd(d, bits=4, device="cpu", seed=SEED)
    iqx, nqx, sqx, gqx = qjl4.quantize(x)
    x_qjl = qjl4.dequantize(iqx, nqx, sqx, gqx).float()
    iqy, nqy, sqy, gqy = qjl4.quantize(y)
    y_qjl = qjl4.dequantize(iqy, nqy, sqy, gqy).float()
    ips_qjl = torch.sum(x_qjl * y_qjl, dim=-1)
    
    # Decompose MSE-only
    alpha_mse = (torch.sum(ips_mse * true_ips) / torch.sum(true_ips**2)).item()
    total_mse_mse = torch.mean((ips_mse - true_ips)**2).item()
    sys_bias_mse = ((alpha_mse - 1.0)**2 * torch.mean(true_ips**2)).item()
    var_mse = torch.mean((ips_mse - alpha_mse * true_ips)**2).item()
    
    # Decompose QJL
    alpha_qjl = (torch.sum(ips_qjl * true_ips) / torch.sum(true_ips**2)).item()
    total_mse_qjl = torch.mean((ips_qjl - true_ips)**2).item()
    sys_bias_qjl = ((alpha_qjl - 1.0)**2 * torch.mean(true_ips**2)).item()
    var_qjl = torch.mean((ips_qjl - alpha_qjl * true_ips)**2).item()
    
    print(f"Dimension {d}:")
    print(f"  4-bit MSE-only:")
    print(f"    Slope (alpha): {alpha_mse:.4f}")
    print(f"    Systematic Bias-squared: {sys_bias_mse:.6f}")
    print(f"    Variance around line:    {var_mse:.6f}")
    print(f"    Total MSE:               {total_mse_mse:.6f}")
    print(f"  4-bit QJL:")
    print(f"    Slope (alpha): {alpha_qjl:.4f} (target 1.0)")
    print(f"    Systematic Bias-squared: {sys_bias_qjl:.6f}")
    print(f"    Variance around line:    {var_qjl:.6f}")
    print(f"    Total MSE:               {total_mse_qjl:.6f}")
    print(f"  Variance difference (QJL - MSE): {var_qjl - var_mse:.6f}")
    print(f"  Bias reduction benefit:          {sys_bias_mse - sys_bias_qjl:.6f}")
    print()
