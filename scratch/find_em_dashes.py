import os
import re

appendices_dir = r"c:\Users\saita\OneDrive\Desktop\AI Everyday\Google Turboquant (Day 1)\kvquant-lab\paper\appendices"
files = [os.path.join(appendices_dir, f) for f in os.listdir(appendices_dir) if f.endswith(".tex")]

for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find all occurrences of double-hyphens, triple-hyphens, or Unicode dashes
    matches = re.finditer(r"-{2,}|—|–", content)
    for m in matches:
        start = m.start()
        # Find line number
        line_num = content[:start].count("\n") + 1
        # Extract surrounding context line
        line_start = content.rfind("\n", 0, start) + 1
        line_end = content.find("\n", start)
        if line_end == -1:
            line_end = len(content)
        line_content = content[line_start:line_end].strip()
        
        # Check if it is a comment
        if line_content.startswith("%"):
            continue
            
        print(f"[{os.path.basename(file_path)}:{line_num}] Match '{m.group(0)}': {line_content}")
