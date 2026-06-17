import json

notebook_path = r"c:\Users\saita\OneDrive\Desktop\AI Everyday\Google Turboquant (Day 1)\kvquant-lab\kaggle_llama_eval.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell containing the bug
found = False
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        source_str = "".join(source)
        if "seq = q_caps[li].shape[0]" in source_str:
            print("Found target cell!")
            # Replace target code
            new_source = []
            for line in source:
                if "seq = q_caps[li].shape[0]" in line:
                    new_source.append("            q_raw = q_caps[li][0]\n")
                    new_source.append("            k_raw = k_caps[li][0]\n")
                    new_source.append("            seq_len = q_raw.shape[0]\n")
                elif "q_h = q_caps[li].view(seq, n_q_heads, d_head).permute(1, 0, 2)" in line:
                    new_source.append("            q_h = q_raw.view(seq_len, n_q_heads, d_head).permute(1, 0, 2)\n")
                elif "k_h = k_caps[li].view(seq, n_kv_heads, d_head).permute(1, 0, 2)" in line:
                    new_source.append("            k_h = k_raw.view(seq_len, n_kv_heads, d_head).permute(1, 0, 2)\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source
            found = True
            break

if found:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Successfully fixed and saved notebook!")
else:
    print("Target cell not found!")
