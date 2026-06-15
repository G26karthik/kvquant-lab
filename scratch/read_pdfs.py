import os
import pypdf

def extract_pdf_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} to {txt_path}...")
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += f"--- PAGE {i+1} ---\n"
        text += page.extract_text() or ""
        text += "\n\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Done. Extracted {len(reader.pages)} pages.")

workspace_dir = "c:/Users/saita/OneDrive/Desktop/AI Everyday/Google Turboquant (Day 1)"
scratch_dir = os.path.join(workspace_dir, "kvquant-lab", "scratch")
os.makedirs(scratch_dir, exist_ok=True)

# Extract TurboQuant PDF
extract_pdf_text(
    os.path.join(workspace_dir, "2504.19874v1.pdf"),
    os.path.join(scratch_dir, "turboquant.txt")
)

# Extract Memory Caching PDF
extract_pdf_text(
    os.path.join(workspace_dir, "MEMORY CACHING RNNS WITH GROWING MEMORY.pdf"),
    os.path.join(scratch_dir, "memory_caching.txt")
)
