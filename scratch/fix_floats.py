import re
import os

sections_dir = r"c:\Users\saita\OneDrive\Desktop\AI Everyday\Google Turboquant (Day 1)\kvquant-lab\paper\sections"
appendices_dir = r"c:\Users\saita\OneDrive\Desktop\AI Everyday\Google Turboquant (Day 1)\kvquant-lab\paper\appendices"

# -------------------------------------------------------------
# 1. Clean results.tex
# -------------------------------------------------------------
results_path = os.path.join(sections_dir, "results.tex")
with open(results_path, "r", encoding="utf-8") as f:
    results_content = f.read()

# Remove external vspaces around tables and figures in results.tex
# Let's replace the specific blocks we changed:
results_content = results_content.replace(
    "\\vspace{-4pt}\n\\begin{table}[ht]",
    "\\begin{table}[ht]\n\\vspace{-4pt}"
)
results_content = results_content.replace(
    "\\end{table}\n\\vspace{-4pt}",
    "\\vspace{-4pt}\n\\end{table}"
)
results_content = results_content.replace(
    "\\vspace{-4pt}\n\\begin{figure}[ht]",
    "\\begin{figure}[ht]\n\\vspace{-4pt}"
)
results_content = results_content.replace(
    "\\end{figure}\n\\vspace{-4pt}",
    "\\vspace{-4pt}\n\\end{figure}"
)

with open(results_path, "w", encoding="utf-8") as f:
    f.write(results_content)
print("Updated results.tex float spacing.")

# -------------------------------------------------------------
# 2. Clean theoretical_validation.tex
# -------------------------------------------------------------
tv_path = os.path.join(sections_dir, "theoretical_validation.tex")
with open(tv_path, "r", encoding="utf-8") as f:
    tv_content = f.read()

tv_content = tv_content.replace(
    "\\vspace{-4pt}\n\\begin{table}[ht]",
    "\\begin{table}[ht]\n\\vspace{-4pt}"
)
tv_content = tv_content.replace(
    "\\end{table}\n\\vspace{-4pt}",
    "\\vspace{-4pt}\n\\end{table}"
)

with open(tv_path, "w", encoding="utf-8") as f:
    f.write(tv_content)
print("Updated theoretical_validation.tex float spacing.")

# -------------------------------------------------------------
# 3. Update all appendix files to [H]
# -------------------------------------------------------------
for filename in os.listdir(appendices_dir):
    if filename.endswith(".tex"):
        path = os.path.join(appendices_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Replace \begin{table}[ht] with \begin{table}[H]
        # and \begin{table}[p] with \begin{table}[H]
        updated_content = content.replace("\\begin{table}[ht]", "\\begin{table}[H]")
        updated_content = updated_content.replace("\\begin{table}[p]", "\\begin{table}[H]")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"Updated {filename} to use [H] floats.")
