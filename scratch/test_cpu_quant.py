import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from turboquant.turbo_quant_demo import TurboQuantMSE as OrigTurboQuantMSE
from turboquant.turbo_quant_demo import TurboQuantProd as OrigTurboQuantProd

try:
    q = OrigTurboQuantMSE(48, bits=4, device="cpu")
    x = torch.randn(10, 48, dtype=torch.float32)
    indices, norms = q.quantize(x)
    recon = q.dequantize(indices, norms)
    print("OrigTurboQuantMSE (QR) works on CPU at d=48! Recon shape:", recon.shape)
except Exception as e:
    print("OrigTurboQuantMSE failed on CPU at d=48:", e)

try:
    qp = OrigTurboQuantProd(48, bits=4, device="cpu")
    x = torch.randn(10, 48, dtype=torch.float32)
    res = qp.quantize(x)
    recon = qp.dequantize(*res)
    print("OrigTurboQuantProd (QR) works on CPU at d=48! Recon shape:", recon.shape)
except Exception as e:
    print("OrigTurboQuantProd failed on CPU at d=48:", e)
