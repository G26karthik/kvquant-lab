import os
import sys
import json
import math
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant.turbo_quant_demo import TurboQuantMSE as OrigTurboQuantMSE
from turboquant.wht_quantizer import TurboQuantMSE as WhtTurboQuantMSE
from turboquant.outlier_channel_quantizer import OutlierChannelQuantizer

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

torch.manual_seed(SEED)
np.random.seed(SEED)

class UniformQuantizer:
    def __init__(self, dim: int, bits: int):
        self.dim = dim
        self.bits = bits
        self.n_levels = 2 ** bits

    def quantize_and_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        # Uniform scalar quantization per vector
        x_f32 = x.float()
        norms = torch.norm(x_f32, dim=-1, keepdim=True)
        x_hat = x_f32 / (norms + 1e-8)
        
        # Determine dynamic range per vector
        max_vals, _ = torch.max(torch.abs(x_hat), dim=-1, keepdim=True)
        scale = max_vals
        step = 2.0 * scale / self.n_levels
        
        # Quantize to [-scale, scale]
        shifted = x_hat + scale
        indices = torch.clamp(torch.round(shifted / (step + 1e-8) - 0.5), 0, self.n_levels - 1)
        recon_hat = (indices + 0.5) * step - scale
        
        return (recon_hat * norms).to(DTYPE)

class LloydMaxNoRotation:
    def __init__(self, dim: int, bits: int, device: str, seed: int = 0):
        self.dim = dim
        self.bits = bits
        self.device = device
        # Use WhtTurboQuantMSE to reuse the Lloyd-Max codebook, but we don't rotate
        self.base_q = WhtTurboQuantMSE(dim, bits, device, seed)

    def quantize_and_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        norms = torch.norm(x_f32, dim=-1, keepdim=True)
        x_hat = x_f32 / (norms + 1e-8)
        
        # Direct scalar quantization without rotation
        indices = torch.bucketize(x_hat, self.base_q.boundaries).to(torch.int8)
        y_hat = self.base_q.codebook[indices.long()]
        
        return (y_hat.to(DTYPE) * norms.to(DTYPE)).half()

def main():
    print("=" * 80)
    print("Running distortion-rate curve benchmark...")
    print("=" * 80)
    
    dim = 64
    num_vectors = 50000
    
    # Generate vectors (isotropic random unit vectors)
    print("Generating 50,000 random unit vectors...")
    x_eval = torch.randn(num_vectors, dim, device=DEVICE)
    x_eval = x_eval / (torch.norm(x_eval, dim=-1, keepdim=True) + 1e-8)
    
    # Also generate query-key pairs for inner product
    x_ip = torch.randn(10000, dim, device=DEVICE)
    x_ip = x_ip / (torch.norm(x_ip, dim=-1, keepdim=True) + 1e-8)
    y_ip = torch.randn(10000, dim, device=DEVICE)
    y_ip = y_ip / (torch.norm(y_ip, dim=-1, keepdim=True) + 1e-8)
    
    bit_levels = [1, 2, 3, 4, 5, 6]
    
    quantizers = [
        "Uniform",
        "Lloyd-Max (No Rotation)",
        "TurboQuant MSE (Flat QR)",
        "WHT + Lloyd-Max",
        "Outlier-split WHT"
    ]
    
    results = {q: [] for q in quantizers}
    shannon_bounds_mse = []
    shannon_bounds_ip = []
    
    for b in bit_levels:
        print(f"\nProcessing bit level b = {b}...")
        
        # Shannon bounds
        shannon_mse = (1.0 / 4.0) ** b
        shannon_ip = (1.0 / dim) * ((1.0 / 4.0) ** b)
        shannon_bounds_mse.append(shannon_mse)
        shannon_bounds_ip.append(shannon_ip)
        
        # 1. Uniform
        q_uni = UniformQuantizer(dim, b)
        recon_uni = q_uni.quantize_and_dequantize(x_eval).float()
        mse_uni = torch.mean(torch.sum((x_eval - recon_uni) ** 2, dim=-1)).item()
        
        recon_uni_ip = q_uni.quantize_and_dequantize(x_ip).float()
        ip_uni = torch.mean((torch.sum(y_ip * x_ip, dim=-1) - torch.sum(y_ip * recon_uni_ip, dim=-1)) ** 2).item()
        
        results["Uniform"].append({"bits": b, "mse": mse_uni, "ip": ip_uni})
        
        # 2. Lloyd-Max (No Rotation)
        q_lm_norot = LloydMaxNoRotation(dim, b, DEVICE, seed=SEED)
        recon_lm = q_lm_norot.quantize_and_dequantize(x_eval).float()
        mse_lm = torch.mean(torch.sum((x_eval - recon_lm) ** 2, dim=-1)).item()
        
        recon_lm_ip = q_lm_norot.quantize_and_dequantize(x_ip).float()
        ip_lm = torch.mean((torch.sum(y_ip * x_ip, dim=-1) - torch.sum(y_ip * recon_lm_ip, dim=-1)) ** 2).item()
        
        results["Lloyd-Max (No Rotation)"].append({"bits": b, "mse": mse_lm, "ip": ip_lm})
        
        # 3. TurboQuant MSE (Flat QR)
        q_tq = OrigTurboQuantMSE(dim, bits=b, device=DEVICE, seed=SEED)
        idx_tq, norm_tq = q_tq.quantize(x_eval.to(DTYPE))
        recon_tq = q_tq.dequantize(idx_tq, norm_tq).float()
        mse_tq = torch.mean(torch.sum((x_eval - recon_tq) ** 2, dim=-1)).item()
        
        idx_tq_ip, norm_tq_ip = q_tq.quantize(x_ip.to(DTYPE))
        recon_tq_ip = q_tq.dequantize(idx_tq_ip, norm_tq_ip).float()
        ip_tq = torch.mean((torch.sum(y_ip * x_ip, dim=-1) - torch.sum(y_ip * recon_tq_ip, dim=-1)) ** 2).item()
        
        results["TurboQuant MSE (Flat QR)"].append({"bits": b, "mse": mse_tq, "ip": ip_tq})
        
        # 4. WHT + Lloyd-Max
        q_wht = WhtTurboQuantMSE(dim, bits=b, device=DEVICE, seed=SEED)
        idx_wht, norm_wht = q_wht.quantize(x_eval.to(DTYPE))
        recon_wht = q_wht.dequantize(idx_wht, norm_wht).float()
        mse_wht = torch.mean(torch.sum((x_eval - recon_wht) ** 2, dim=-1)).item()
        
        idx_wht_ip, norm_wht_ip = q_wht.quantize(x_ip.to(DTYPE))
        recon_wht_ip = q_wht.dequantize(idx_wht_ip, norm_wht_ip).float()
        ip_wht = torch.mean((torch.sum(y_ip * x_ip, dim=-1) - torch.sum(y_ip * recon_wht_ip, dim=-1)) ** 2).item()
        
        results["WHT + Lloyd-Max"].append({"bits": b, "mse": mse_wht, "ip": ip_wht})
        
        # 5. Outlier-split WHT
        q_out = OutlierChannelQuantizer(dim, avg_bits=float(b), device=DEVICE, seed=SEED)
        q_out.calibrate(x_eval)
        comp_out = q_out.quantize(x_eval.to(DTYPE))
        recon_out = q_out.dequantize(comp_out).float()
        mse_out = torch.mean(torch.sum((x_eval - recon_out) ** 2, dim=-1)).item()
        
        comp_out_ip = q_out.quantize(x_ip.to(DTYPE))
        recon_out_ip = q_out.dequantize(comp_out_ip).float()
        ip_out = torch.mean((torch.sum(y_ip * x_ip, dim=-1) - torch.sum(y_ip * recon_out_ip, dim=-1)) ** 2).item()
        
        results["Outlier-split WHT"].append({"bits": b, "mse": mse_out, "ip": ip_out})
        
        # Print summaries
        print(f"  Uniform:     MSE={mse_uni:.6f}, IP={ip_uni:.8f}")
        print(f"  Lloyd-Max:   MSE={mse_lm:.6f}, IP={ip_lm:.8f}")
        print(f"  TurboQuant:  MSE={mse_tq:.6f}, IP={ip_tq:.8f}")
        print(f"  WHT+LM:      MSE={mse_wht:.6f}, IP={ip_wht:.8f}")
        print(f"  Outlier-split: MSE={mse_out:.6f}, IP={ip_out:.8f}")
        
    # Build export JSON
    out_json = {
        "bit_levels": bit_levels,
        "shannon_bounds_mse": shannon_bounds_mse,
        "shannon_bounds_ip": shannon_bounds_ip,
        "results": results
    }
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "distortion_rate.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved JSON results to {json_path}")
    
    html_path = os.path.join(results_dir, "distortion_rate.html")
    generate_html(out_json, html_path)
    print(f"Saved HTML report to {html_path}")

def generate_html(data, output_path):
    import json
    json_data_str = json.dumps(data)
    
    # Non-f-string to prevent brace conflicts
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Distortion-Rate Curves</title>
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
    max-width: 1100px;
    margin: 0 auto;
  }
  header {
    text-align: center;
    margin-bottom: 40px;
  }
  header h1 {
    font-size: 28px;
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
  .card {
    background: #1a1d28;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 30px;
  }
  .card h2 {
    font-size: 16px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 20px;
  }
  .charts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 20px;
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
    <h1>Distortion-Rate Curve Evaluation Dashboard</h1>
    <div class="subtitle">Comparing 5 Quantizer Configurations Across Bit Levels 1-6 &middot; d=64</div>
  </header>

  <div class="charts-grid">
    <div class="card">
      <h2>MSE Distortion (y-axis log-scale) vs Bit-rate (x-axis)</h2>
      <div class="chart-container" id="mse-chart">
        <svg id="svg-mse" width="100%" height="100%" viewBox="0 0 500 350"></svg>
      </div>
    </div>
    <div class="card">
      <h2>Inner Product (IP) Distortion (y-axis log-scale) vs Bit-rate (x-axis)</h2>
      <div class="chart-container" id="ip-chart">
        <svg id="svg-ip" width="100%" height="100%" viewBox="0 0 500 350"></svg>
      </div>
    </div>
  </div>

  <div class="footer">
    Generated by distortion_rate_curve.py &middot; d=64 &middot; Seed: 42 &middot; Device: DEVICE_HOLDER
  </div>
</div>

<script>
const rawData = JSON_DATA_HOLDER;

function drawChart(svgId, metric) {
    const svg = document.getElementById(svgId);
    const width = 500;
    const height = 350;
    const padding = { top: 30, right: 150, bottom: 40, left: 60 };
    
    // Clear existing contents
    svg.innerHTML = '';
    
    const bitLevels = rawData.bit_levels;
    const quantizers = Object.keys(rawData.results);
    
    // Find min and max for log scale
    let valMin = Infinity;
    let valMax = -Infinity;
    
    // Scan rawData results
    quantizers.forEach(q => {
        rawData.results[q].forEach(pt => {
            const val = pt[metric];
            if (val > 0 && val < valMin) valMin = val;
            if (val > valMax) valMax = val;
        });
    });
    
    // Also scan Shannon bounds
    const lbArray = metric === 'mse' ? rawData.shannon_bounds_mse : rawData.shannon_bounds_ip;
    lbArray.forEach(val => {
        if (val > 0 && val < valMin) valMin = val;
        if (val > valMax) valMax = val;
    });
    
    // Add margin to min/max
    valMin = valMin * 0.5;
    valMax = valMax * 1.5;
    
    const logMin = Math.log10(valMin);
    const logMax = Math.log10(valMax);
    
    // X scale: 1 to 6
    function getX(b) {
        return padding.left + ((b - 1) / 5) * (width - padding.left - padding.right);
    }
    
    // Y scale (log scale)
    function getY(val) {
        const logVal = Math.log10(val);
        return padding.top + (1 - (logVal - logMin) / (logMax - logMin)) * (height - padding.top - padding.bottom);
    }
    
    // Draw background grid lines (Y axis - powers of 10)
    const startPower = Math.ceil(logMin);
    const endPower = Math.floor(logMax);
    for (let p = startPower; p <= endPower; p++) {
        const yVal = Math.pow(10, p);
        const yPos = getY(yVal);
        // Grid line
        const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
        grid.setAttribute("x1", padding.left);
        grid.setAttribute("y1", yPos);
        grid.setAttribute("x2", width - padding.right);
        grid.setAttribute("y2", yPos);
        grid.setAttribute("stroke", "#2a2d3a");
        grid.setAttribute("stroke-width", "1");
        svg.appendChild(grid);
        
        // Label
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", padding.left - 8);
        text.setAttribute("y", yPos + 4);
        text.setAttribute("fill", "#8b8fa3");
        text.setAttribute("font-size", "9");
        text.setAttribute("text-anchor", "end");
        text.textContent = "10^" + p;
        svg.appendChild(text);
    }
    
    // Draw grid lines (X axis)
    bitLevels.forEach(b => {
        const xPos = getX(b);
        const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
        grid.setAttribute("x1", xPos);
        grid.setAttribute("y1", padding.top);
        grid.setAttribute("x2", xPos);
        grid.setAttribute("y2", height - padding.bottom);
        grid.setAttribute("stroke", "#2a2d3a");
        grid.setAttribute("stroke-width", "1");
        svg.appendChild(grid);
        
        // Label
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", xPos);
        text.setAttribute("y", height - padding.bottom + 15);
        text.setAttribute("fill", "#8b8fa3");
        text.setAttribute("font-size", "10");
        text.setAttribute("text-anchor", "middle");
        text.textContent = b + "b";
        svg.appendChild(text);
    });
    
    // Draw Shannon bound line
    let lbPathD = "";
    bitLevels.forEach((b, idx) => {
        const x = getX(b);
        const y = getY(lbArray[idx]);
        lbPathD += (idx === 0 ? "M" : "L") + x + " " + y;
    });
    const lbPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    lbPath.setAttribute("d", lbPathD);
    lbPath.setAttribute("fill", "none");
    lbPath.setAttribute("stroke", "#8b8fa3");
    lbPath.setAttribute("stroke-width", "2");
    lbPath.setAttribute("stroke-dasharray", "4,4");
    svg.appendChild(lbPath);
    
    // Colors
    const colors = {
        "Uniform": "#ff5555",
        "Lloyd-Max (No Rotation)": "#ffb86c",
        "TurboQuant MSE (Flat QR)": "#50fa7b",
        "WHT + Lloyd-Max": "#8be9fd",
        "Outlier-split WHT": "#bd93f9"
    };
    
    // Draw curves for each quantizer
    quantizers.forEach(q => {
        const pts = rawData.results[q];
        let pathD = "";
        pts.forEach((pt, idx) => {
            const x = getX(pt.bits);
            const y = getY(pt[metric]);
            pathD += (idx === 0 ? "M" : "L") + x + " " + y;
        });
        
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", pathD);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", colors[q]);
        path.setAttribute("stroke-width", "2");
        svg.appendChild(path);
        
        // Draw dots
        pts.forEach(pt => {
            const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            circle.setAttribute("cx", getX(pt.bits));
            circle.setAttribute("cy", getY(pt[metric]));
            circle.setAttribute("r", "3.5");
            circle.setAttribute("fill", colors[q]);
            svg.appendChild(circle);
        });
    });
    
    // Draw legend
    let legY = padding.top;
    
    // Add Shannon legend item
    const lbLegLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    lbLegLine.setAttribute("x1", width - padding.right + 15);
    lbLegLine.setAttribute("y1", legY + 5);
    lbLegLine.setAttribute("x2", width - padding.right + 35);
    lbLegLine.setAttribute("y2", legY + 5);
    lbLegLine.setAttribute("stroke", "#8b8fa3");
    lbLegLine.setAttribute("stroke-width", "2");
    lbLegLine.setAttribute("stroke-dasharray", "3,3");
    svg.appendChild(lbLegLine);
    
    const lbLegText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    lbLegText.setAttribute("x", width - padding.right + 40);
    lbLegText.setAttribute("y", legY + 9);
    lbLegText.setAttribute("fill", "#e6e6e6");
    lbLegText.setAttribute("font-size", "10");
    lbLegText.textContent = "Shannon LB";
    svg.appendChild(lbLegText);
    
    legY += 20;
    
    quantizers.forEach(q => {
        const legLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
        legLine.setAttribute("x1", width - padding.right + 15);
        legLine.setAttribute("y1", legY + 5);
        legLine.setAttribute("x2", width - padding.right + 35);
        legLine.setAttribute("y2", legY + 5);
        legLine.setAttribute("stroke", colors[q]);
        legLine.setAttribute("stroke-width", "2");
        svg.appendChild(legLine);
        
        const legText = document.createElementNS("http://www.w3.org/2000/svg", "text");
        legText.setAttribute("x", width - padding.right + 40);
        legText.setAttribute("y", legY + 9);
        legText.setAttribute("fill", "#e6e6e6");
        legText.setAttribute("font-size", "10");
        legText.textContent = q;
        svg.appendChild(legText);
        
        legY += 20;
    });
    
    // Axes labels
    const yLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    yLabel.setAttribute("transform", "rotate(-90)");
    yLabel.setAttribute("x", -(padding.top + (height - padding.top - padding.bottom)/2));
    yLabel.setAttribute("y", padding.left - 42);
    yLabel.setAttribute("fill", "#8b8fa3");
    yLabel.setAttribute("font-size", "10");
    yLabel.setAttribute("text-anchor", "middle");
    yLabel.textContent = metric === 'mse' ? "Mean Squared Error (MSE) Distortion" : "Inner Product (IP) Distortion";
    svg.appendChild(yLabel);
    
    const xLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    xLabel.setAttribute("x", padding.left + (width - padding.left - padding.right)/2);
    xLabel.setAttribute("y", height - 5);
    xLabel.setAttribute("fill", "#8b8fa3");
    xLabel.setAttribute("font-size", "10");
    xLabel.setAttribute("text-anchor", "middle");
    xLabel.textContent = "Bit Rate (bits per channel)";
    svg.appendChild(xLabel);
}

// Draw both charts
drawChart('svg-mse', 'mse');
drawChart('svg-ip', 'ip');
</script>
</body>
</html>"""
    
    final_html = html_template.replace("JSON_DATA_HOLDER", json_data_str).replace("DEVICE_HOLDER", DEVICE.upper())
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)

if __name__ == "__main__":
    main()
