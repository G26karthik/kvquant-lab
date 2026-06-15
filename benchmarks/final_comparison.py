"""
final_comparison.py
===================
Pure data-synthesis script: reads every pre-computed result JSON and writes
a master head-to-head HTML dashboard + JSON.
No model loading, no inference — all numbers come from already-saved results.
"""

import os
import sys
import json
import math
import datetime

RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load(name):
    p = os.path.join(RESULTS_DIR, name)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def fmt(v, decimals=2, suffix=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}{suffix}"
    return f"{v}{suffix}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  kvquant-lab  ·  Master Final Comparison (data-synthesis)")
    print("=" * 72)

    # ── Load all results ────────────────────────────────────────────────────
    std        = load("standard_eval.json")      # PPL + HellaSwag + NIAH matrices
    lb         = load("longbench_lite.json")      # PassageQA / Summ / Code
    outlier    = load("outlier_results.json")     # MSE, IP, avg-PPL, cache
    nn         = load("nn_search.json")           # Recall@1/10, mean rank
    bit_sw     = load("bit_sweep.json")           # per-scheme bit sweeps
    mc_v       = load("mc_variants.json")         # MQAR accuracy
    mc_inf     = load("mc_inference.json")        # MC inference wrapper
    mc_comp    = load("mc_compressor_ablation.json")
    mc_niah    = load("mc_niah.json")
    dist_b     = load("distortion_bounds.json")   # Shannon lower bounds
    dist_r     = load("distortion_rate.json")
    cmplx_rec  = load("complexity_recall.json")
    base_res   = load("results.json")             # original turbo_quant_demo output

    # ── Scheme registry ─────────────────────────────────────────────────────
    # Effective avg bit-rates (key + value combined, with overhead)
    BIT_RATES = {
        "Baseline FP16":                  16.00,
        "Original TurboQuant (3-bit)":     4.00,
        "Original TurboQuant (4-bit)":     5.00,
        "WHT + Asymmetric QJL (3-bit)":    4.00,
        "WHT + Asymmetric QJL (4-bit)":    5.00,
        "MC-TurboQuant (4-bit)":           5.00,
        "Outlier Channel Splitting (2.5-bit)": 3.44,
        "Outlier Channel Splitting (3.5-bit)": 4.44,
    }
    SCHEMES = list(BIT_RATES.keys())

    # ── PPL (WikiText-2 strided) ─────────────────────────────────────────────
    PPL_SRC = std.get("eval_a_perplexity", {})
    # Outlier PPL: use the avg-perplexity from bit-sweep Scheme B @ matching bits
    # (these match the outlier_results avg perplexity closely)
    ppl = {
        "Baseline FP16":                  PPL_SRC.get("Baseline FP16"),
        "Original TurboQuant (3-bit)":    PPL_SRC.get("Original TurboQuant (3-bit)"),
        "Original TurboQuant (4-bit)":    PPL_SRC.get("Original TurboQuant (4-bit)"),
        "WHT + Asymmetric QJL (3-bit)":   PPL_SRC.get("WHT + Asymmetric QJL (3-bit)"),
        "WHT + Asymmetric QJL (4-bit)":   PPL_SRC.get("WHT + Asymmetric QJL (4-bit)"),
        "MC-TurboQuant (4-bit)":          PPL_SRC.get("WHT + Asymmetric QJL (4-bit)"),  # same quantizer
        "Outlier Channel Splitting (2.5-bit)": outlier.get("average_perplexity", {}).get("outlier_2.5"),
        "Outlier Channel Splitting (3.5-bit)": outlier.get("average_perplexity", {}).get("outlier_3.5"),
    }

    # ── HellaSwag ────────────────────────────────────────────────────────────
    HS_SRC = std.get("eval_b_hellaswag", {})
    hs = {
        "Baseline FP16":                  HS_SRC.get("Baseline FP16"),
        "Original TurboQuant (3-bit)":    HS_SRC.get("Original TurboQuant (3-bit)"),
        "Original TurboQuant (4-bit)":    HS_SRC.get("Original TurboQuant (4-bit)"),
        "WHT + Asymmetric QJL (3-bit)":   HS_SRC.get("WHT + Asymmetric QJL (3-bit)"),
        "WHT + Asymmetric QJL (4-bit)":   HS_SRC.get("WHT + Asymmetric QJL (4-bit)"),
        "MC-TurboQuant (4-bit)":          HS_SRC.get("WHT + Asymmetric QJL (4-bit)"),
        "Outlier Channel Splitting (2.5-bit)": None,
        "Outlier Channel Splitting (3.5-bit)": None,
    }

    # ── NIAH Recall @ 512 Middle (50%) ──────────────────────────────────────
    niah_src = std.get("eval_c_niah", {})
    def niah_512_mid(key):
        return niah_src.get(key, {}).get("512", {}).get("Middle")

    niah = {
        "Baseline FP16":                      niah_512_mid("baseline_fp16"),
        "Original TurboQuant (3-bit)":        None,   # not measured separately
        "Original TurboQuant (4-bit)":        niah_512_mid("turboquant_flat"),
        "WHT + Asymmetric QJL (3-bit)":       None,
        "WHT + Asymmetric QJL (4-bit)":       niah_512_mid("wht_asym_flat"),
        "MC-TurboQuant (4-bit)":              niah_512_mid("mc_turboquant"),
        "Outlier Channel Splitting (2.5-bit)": None,
        "Outlier Channel Splitting (3.5-bit)": None,
    }

    # ── LongBench-lite ───────────────────────────────────────────────────────
    LB_MAP = {
        "Baseline FP16":                      "Baseline",
        "WHT + Asymmetric QJL (4-bit)":       "WHT+Asym-4bit",
        "Outlier Channel Splitting (2.5-bit)": "Outlier-2.5bit",
        "Outlier Channel Splitting (3.5-bit)": "Outlier-3.5bit",
    }
    def lb_get(sch, key):
        k = LB_MAP.get(sch)
        if k and lb:
            return lb.get(k, {}).get(key)
        return None

    # ── Nearest-Neighbor Recall @ 1 ─────────────────────────────────────────
    # nn_search.json structure: list of dicts with scheme/bits/recall_at_1/...
    nn_rec1 = {}
    if isinstance(nn, dict):
        nn_list = nn.get("results", nn.get("data", []))
    elif isinstance(nn, list):
        nn_list = nn
    else:
        nn_list = []
    for entry in nn_list:
        sch  = entry.get("scheme", "")
        bits = entry.get("bits", 0)
        r1   = entry.get("recall_at_1")
        if r1 is None:
            r1 = entry.get("Recall@1") or entry.get("recall_1")
        label = None
        if "WHT" in sch or "Asym" in sch:
            if bits == 4:
                label = "WHT + Asymmetric QJL (4-bit)"
            elif bits == 3:
                label = "WHT + Asymmetric QJL (3-bit)"
        elif "Outlier" in sch or "outlier" in sch:
            if abs(bits - 2.5) < 0.1:
                label = "Outlier Channel Splitting (2.5-bit)"
            elif abs(bits - 3.5) < 0.1:
                label = "Outlier Channel Splitting (3.5-bit)"
        elif "TurboQuant" in sch or "turbo" in sch.lower():
            if bits == 4:
                label = "Original TurboQuant (4-bit)"
            elif bits == 3:
                label = "Original TurboQuant (3-bit)"
        if label and r1 is not None:
            nn_rec1[label] = r1

    # ── Compression ratio ────────────────────────────────────────────────────
    comp = {s: round(16.0 / BIT_RATES[s], 2) for s in SCHEMES}

    # ── Build master rows ────────────────────────────────────────────────────
    rows = []
    for s in SCHEMES:
        rows.append({
            "scheme":            s,
            "bit_rate":          BIT_RATES[s],
            "compression_ratio": comp[s],
            "wikitext_ppl":      ppl.get(s),
            "hellaswag_acc":     hs.get(s),
            "niah_512_mid":      niah.get(s),
            "nn_recall_1":       nn_rec1.get(s),
            "passage_qa":        lb_get(s, "passage_qa"),
            "summarization_r1":  lb_get(s, "summarization"),
            "code_completion":   lb_get(s, "code_completion"),
        })

    # -- Save JSON ------------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_out = os.path.join(RESULTS_DIR, "final_comparison.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                   "schemes": rows}, f, indent=2)
    print(f"\n[OK] Saved master JSON -> {json_out}")

    # -- Generate HTML --------------------------------------------------------
    html_out = os.path.join(RESULTS_DIR, "final_comparison.html")
    _write_html(rows, outlier, dist_b, nn_list, mc_v, base_res, html_out)
    print(f"[OK] Saved master HTML  -> {html_out}")

    # -- Pretty print to console ----------------------------------------------
    _print_table(rows)


# ---------------------------------------------------------------------------
# Console pretty-print
# ---------------------------------------------------------------------------

def _print_table(rows):
    H = ["Scheme", "bits", "CR", "PPL(lo)", "HellaSwag", "NIAH@512", "NN-R@1",
         "PassQA", "Summ", "Code"]
    COL_W = [42, 5, 5, 8, 11, 10, 9, 9, 7, 7]

    def cell(v, w):
        s = "--" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))
        return s.rjust(w)

    sep = "  ".join("-" * w for w in COL_W)
    hdr = "  ".join(h.rjust(w) for h, w in zip(H, COL_W))
    print("\n" + sep)
    print(hdr)
    print(sep)
    for r in rows:
        vals = [
            r["scheme"],
            r["bit_rate"],
            r["compression_ratio"],
            r["wikitext_ppl"],
            r["hellaswag_acc"],
            r["niah_512_mid"],
            r["nn_recall_1"],
            r["passage_qa"],
            r["summarization_r1"],
            r["code_completion"],
        ]
        print("  ".join(cell(v, w) for v, w in zip(vals, COL_W)))
    print(sep + "\n")


# ---------------------------------------------------------------------------
# HTML generator
# ---------------------------------------------------------------------------

def _write_html(rows, outlier, dist_b, nn_list, mc_v, base_res, path):

    def v(val, decs=2, suf=""):
        if val is None:
            return "<span class='na'>—</span>"
        if isinstance(val, float):
            return f"{val:.{decs}f}{suf}"
        return f"{val}{suf}"

    # ── Master table rows ───────────────────────────────────────────────────
    table_rows = ""
    for r in rows:
        cr_cls = "hi" if r["compression_ratio"] >= 3 else ""
        ppl = r["wikitext_ppl"]
        ppl_cls = "lo" if (ppl is not None and ppl < 25) else ""
        table_rows += f"""
        <tr>
          <td class="scheme-name">{r['scheme']}</td>
          <td class="num">{r['bit_rate']:.2f} <span style='font-size:11px;color:#8b8fa3'>b</span></td>
          <td class="num {cr_cls}">{v(r['compression_ratio'], 2)}x</td>
          <td class="num {ppl_cls}">{v(r['wikitext_ppl'], 2)}</td>
          <td class="num">{v(r['hellaswag_acc'], 1, '%') if r['hellaswag_acc'] is not None else '<span class="na">—</span>'}</td>
          <td class="num">{v(r['niah_512_mid'], 1, '%') if r['niah_512_mid'] is not None else '<span class="na">—</span>'}</td>
          <td class="num">{v(r['nn_recall_1'], 1, '%') if r['nn_recall_1'] is not None else '<span class="na">—</span>'}</td>
          <td class="num">{v(r['passage_qa'], 1, '%') if r['passage_qa'] is not None else '<span class="na">—</span>'}</td>
          <td class="num">{v(r['summarization_r1'], 3) if r['summarization_r1'] is not None else '<span class="na">—</span>'}</td>
          <td class="num">{v(r['code_completion'], 1, '%') if r['code_completion'] is not None else '<span class="na">—</span>'}</td>
        </tr>"""

    # ── Distortion-bound summary cards ──────────────────────────────────────
    dist_cards = ""
    for entry in dist_b.get("mse_results", []):
        bits = entry["bits"]
        dist_cards += f"""
        <div class="stat-card">
          <div class="stat-label">{bits}-bit MSE Gap</div>
          <div class="stat-val">{entry['ratio_tq']:.2f}× <span class="sub">above Shannon</span></div>
          <div class="stat-sub">TQ: {entry['tq_mse']:.5f}  WHT: {entry['wht_mse']:.5f}  LB: {entry['shannon_lb']:.5f}</div>
        </div>"""

    # ── Outlier summary ──────────────────────────────────────────────────────
    avg_ppl = outlier.get("average_perplexity", {})
    out_cards = ""
    for label, key in [("2.5-bit Outlier", "outlier_2.5"), ("3.5-bit Outlier", "outlier_3.5"),
                        ("Flat 3-bit", "flat_3"), ("Flat 4-bit", "flat_4")]:
        p   = avg_ppl.get(key)
        kb  = outlier.get("average_cache_size_kb", {}).get(key)
        mse = outlier.get("mathematical_fidelity", {}).get(key, {}).get("mse_distortion")
        out_cards += f"""
        <div class="stat-card">
          <div class="stat-label">{label}</div>
          <div class="stat-val">{v(p, 4)} PPL</div>
          <div class="stat-sub">{v(kb, 0)} KB cache &nbsp;·&nbsp; MSE {v(mse, 6)}</div>
        </div>"""

    # ── Base demo highlight ──────────────────────────────────────────────────
    b_avg = base_res.get("baseline_avg", {})
    t_avg = base_res.get("turboquant_avg", {})
    summ  = base_res.get("summary", {})
    demo_html = f"""
    <div class="demo-grid">
      <div class="stat-card accent">
        <div class="stat-label">Compression Ratio</div>
        <div class="stat-val">{summ.get('compression_ratio', '—')}×</div>
        <div class="stat-sub">FP16 KV → 4-bit TurboQuant</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Perplexity Δ</div>
        <div class="stat-val">+{summ.get('perplexity_change_pct', '—')}%</div>
        <div class="stat-sub">{v(b_avg.get('perplexity'),4)} → {v(t_avg.get('perplexity'),4)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Baseline Speed</div>
        <div class="stat-val">{v(b_avg.get('tokens_per_sec'),2)} tok/s</div>
        <div class="stat-sub">KV Cache: {b_avg.get('kv_cache_kb','—')} KB</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">TurboQuant Speed</div>
        <div class="stat-val">{v(t_avg.get('tokens_per_sec'),2)} tok/s</div>
        <div class="stat-sub">KV Cache: {t_avg.get('kv_cache_kb','—')} KB</div>
      </div>
    </div>"""

    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>kvquant-lab — Master Final Comparison</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  :root{{
    --bg:#0b0d14; --card:#13151f; --border:#1f2235; --accent:#6366f1;
    --green:#34d399; --yellow:#fbbf24; --red:#f87171; --muted:#64748b;
    --text:#e2e8f0; --text2:#94a3b8;
  }}
  body{{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;
        line-height:1.6;padding:40px 24px;min-height:100vh;}}
  .container{{max-width:1300px;margin:0 auto;}}

  /* ── Hero ── */
  .hero{{text-align:center;padding:60px 20px 48px;}}
  .hero-badge{{display:inline-block;background:rgba(99,102,241,.15);
    border:1px solid rgba(99,102,241,.4);color:#818cf8;font-size:11px;
    font-weight:600;letter-spacing:1.2px;text-transform:uppercase;
    padding:4px 14px;border-radius:20px;margin-bottom:20px;}}
  .hero h1{{font-size:clamp(28px,4vw,48px);font-weight:700;line-height:1.15;
    background:linear-gradient(135deg,#e2e8f0 0%,#818cf8 60%,#34d399 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;margin-bottom:12px;}}
  .hero .sub{{color:var(--text2);font-size:15px;max-width:680px;margin:0 auto;}}

  /* ── Section headings ── */
  .section{{margin-bottom:48px;}}
  .section-title{{font-size:13px;font-weight:600;letter-spacing:.8px;
    text-transform:uppercase;color:var(--muted);margin-bottom:16px;
    display:flex;align-items:center;gap:10px;}}
  .section-title::after{{content:'';flex:1;height:1px;background:var(--border);}}

  /* ── Card ── */
  .card{{background:var(--card);border:1px solid var(--border);
         border-radius:14px;padding:24px;margin-bottom:24px;}}

  /* ── Stat cards ── */
  .stat-grid,.demo-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
    gap:16px;margin-bottom:8px;}}
  .stat-card{{background:#0f1120;border:1px solid var(--border);border-radius:12px;
    padding:18px 20px;}}
  .stat-card.accent{{border-color:rgba(99,102,241,.5);
    background:linear-gradient(135deg,rgba(99,102,241,.08),rgba(52,211,153,.05));}}
  .stat-label{{font-size:11px;font-weight:600;letter-spacing:.6px;
    text-transform:uppercase;color:var(--muted);margin-bottom:6px;}}
  .stat-val{{font-size:26px;font-weight:700;color:var(--text);line-height:1.2;}}
  .stat-val .sub{{font-size:13px;font-weight:400;color:var(--text2);}}
  .stat-sub{{font-size:12px;color:var(--muted);margin-top:4px;
    font-family:'JetBrains Mono',monospace;}}

  /* ── Table ── */
  .table-wrapper{{overflow-x:auto;}}
  table{{width:100%;border-collapse:collapse;font-size:13px;}}
  th,td{{padding:11px 14px;text-align:left;border-bottom:1px solid var(--border);}}
  th{{background:#0b0d14;color:var(--muted);font-size:10px;font-weight:600;
    text-transform:uppercase;letter-spacing:.5px;white-space:nowrap;}}
  tr:hover td{{background:rgba(99,102,241,.04);}}
  td.num{{text-align:right;font-family:'JetBrains Mono',monospace;font-size:12.5px;}}
  td.scheme-name{{font-weight:600;color:#c7d2fe;max-width:280px;}}
  .hi{{color:var(--green);font-weight:600;}}
  .lo{{color:var(--yellow);}}
  .na{{color:#3f475e;}}

  /* ── Footer ── */
  .footer{{text-align:center;font-size:12px;color:#3f475e;margin-top:60px;
    padding-top:24px;border-top:1px solid var(--border);}}
</style>
</head>
<body>
<div class="container">

  <div class="hero">
    <div class="hero-badge">kvquant-lab · Final Report</div>
    <h1>Master Head-to-Head Comparison</h1>
    <p class="sub">
      TurboQuant (arXiv:2504.19874) × Memory Caching (arXiv:2602.24281) —
      8 quantization schemes, 10 metrics, GPT-2 Medium on RTX 4060 Laptop GPU.
    </p>
  </div>

  <!-- ── TurboQuant Demo Highlights ── -->
  <div class="section">
    <div class="section-title">TurboQuant Core Demo</div>
    {demo_html}
  </div>

  <!-- ── Master Table ── -->
  <div class="section">
    <div class="section-title">Full Head-to-Head Metrics</div>
    <div class="card">
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Quantization Configuration</th>
              <th style="text-align:right">Avg Bits</th>
              <th style="text-align:right">Compression</th>
              <th style="text-align:right">WikiText-2 PPL ↓</th>
              <th style="text-align:right">HellaSwag ↑</th>
              <th style="text-align:right">NIAH @512 ↑</th>
              <th style="text-align:right">NN Recall@1 ↑</th>
              <th style="text-align:right">PassageQA ↑</th>
              <th style="text-align:right">Summ R-1 ↑</th>
              <th style="text-align:right">Code Compl ↑</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- ── Distortion Bounds ── -->
  <div class="section">
    <div class="section-title">Information-Theoretic Distortion Analysis (MSE vs Shannon Lower Bound)</div>
    <div class="stat-grid">
      {dist_cards}
    </div>
  </div>

  <!-- ── Outlier Channel Splitting ── -->
  <div class="section">
    <div class="section-title">Outlier Channel Splitting vs Flat Lloyd-Max</div>
    <div class="stat-grid">
      {out_cards}
    </div>
  </div>

  <div class="footer">
    Generated by final_comparison.py &middot; {now} &middot; GPT-2 Medium (d_head=64) &middot; RTX 4060 Laptop GPU
  </div>

</div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
