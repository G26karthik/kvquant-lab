import os
import sys
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from turboquant.turbo_quant_demo import TurboQuantMSE, TurboQuantProd

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
N_pairs = 10000
rho = 0.8

for d in [16, 32, 48, 64, 96, 128, 192, 256]:
    # Generate correlated vectors
    x = torch.randn(N_pairs, d, dtype=torch.float32)
    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    
    z = torch.randn(N_pairs, d, dtype=torch.float32)
    z = z / (torch.norm(z, dim=-1, keepdim=True) + 1e-8)
    
    y = rho * x + math.sqrt(1 - rho**2) * z
    y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
    
    true_ips = torch.sum(x * y, dim=-1)
    
    # 4-bit MSE
    mse4 = TurboQuantMSE(d, bits=4, device="cpu", seed=SEED)
    ix, nx = mse4.quantize(x)
    x_hat = mse4.dequantize(ix, nx).float()
    iy, ny = mse4.quantize(y)
    y_hat = mse4.dequantize(iy, ny).float()
    ips_mse = torch.sum(x_hat * y_hat, dim=-1)
    
    # 4-bit QJL (3-bit MSE + 1-bit QJL)
    qjl4 = TurboQuantProd(d, bits=4, device="cpu", seed=SEED)
    iqx, nqx, sqx, gqx = qjl4.quantize(x)
    x_qjl = qjl4.dequantize(iqx, nqx, sqx, gqx).float()
    iqy, nqy, sqy, gqy = qjl4.quantize(y)
    y_qjl = qjl4.dequantize(iqy, nqy, sqy, gqy).float()
    ips_qjl = torch.sum(x_qjl * y_qjl, dim=-1)
    
    # Total MSE
    total_mse_mse4 = torch.mean((ips_mse - true_ips)**2).item()
    total_mse_qjl = torch.mean((ips_qjl - true_ips)**2).item()
    
    # Decompose MSE-only
    alpha_mse = (torch.sum(ips_mse * true_ips) / torch.sum(true_ips**2)).item()
    sys_bias_mse = ((alpha_mse - 1.0)**2 * torch.mean(true_ips**2)).item()
    var_mse = torch.mean((ips_mse - alpha_mse * true_ips)**2).item()
    
    # Decompose QJL
    alpha_qjl = (torch.sum(ips_qjl * true_ips) / torch.sum(true_ips**2)).item()
    sys_bias_qjl = ((alpha_qjl - 1.0)**2 * torch.mean(true_ips**2)).item()
    var_qjl = torch.mean((ips_qjl - alpha_qjl * true_ips)**2).item()
    
    print(f"Dimension {d}:")
    print(f"  4-bit MSE-only: Total MSE = {total_mse_mse4:.6f} (Sys Bias-sq = {sys_bias_mse:.6f}, Var = {var_mse:.6f})")
    print(f"  4-bit QJL:      Total MSE = {total_mse_qjl:.6f} (Sys Bias-sq = {sys_bias_qjl:.6f}, Var = {var_qjl:.6f})")
    print(f"  QJL is better than 4-bit MSE: {total_mse_qjl < total_mse_mse4}")
    print(f"  Added Var of QJL (vs 4-bit MSE): {var_qjl - var_mse:.6f}")
    print(f"  Bias reduction benefit:           {sys_bias_mse - sys_bias_qjl:.6f}")
    print()
