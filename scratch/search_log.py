import os

dirs = [
    r"c:\Users\saita\OneDrive\Desktop\AI Everyday\Google Turboquant (Day 1)\kvquant-lab\paper\sections",
    r"c:\Users\saita\OneDrive\Desktop\AI Everyday\Google Turboquant (Day 1)\kvquant-lab\paper\appendices",
    r"c:\Users\saita\OneDrive\Desktop\AI Everyday\Google Turboquant (Day 1)\kvquant-lab\paper"
]

for d in dirs:
    for filename in os.listdir(d):
        if filename.endswith(".tex"):
            filepath = os.path.join(d, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            if "includegraphics" in content:
                print(f"Found includegraphics in {filename}")
            if "figures/" in content:
                print(f"Found figures/ reference in {filename}")
