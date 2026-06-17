import os
import re

pdf_path = r"c:\Users\saita\OneDrive\Desktop\AI Everyday\Google Turboquant (Day 1)\kvquant-lab\paper\main.pdf"
if not os.path.exists(pdf_path):
    print("PDF not found!")
    exit(1)

with open(pdf_path, "rb") as f:
    content = f.read()

# Find all occurrences of /Type /Page (case-sensitive or insensitive)
pages = re.findall(b"/Type\s*/Page", content)
print("Page count via /Type /Page regex:", len(pages))

# Also try /Count in Catalog/Pages
counts = re.findall(b"/Count\s+(\d+)", content)
if counts:
    print("Page counts found in PDF metadata:", [int(c) for c in counts])
