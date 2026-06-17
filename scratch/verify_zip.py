import zipfile
import os

zip_path = r"c:\Users\saita\OneDrive\Desktop\AI Everyday\Google Turboquant (Day 1)\kvquant-lab\paper\arxiv_submission.zip"

if not os.path.exists(zip_path):
    print("Error: ZIP file not found!")
    exit(1)

print("Verifying ZIP file contents...")
with zipfile.ZipFile(zip_path, 'r') as zipf:
    namelist = zipf.namelist()
    
    # 1. Check if main.tex is in root
    if "main.tex" in namelist:
        print("[PASS] main.tex is at the root directory of the ZIP.")
    else:
        print("[FAIL] main.tex is NOT at the root directory of the ZIP.")
        
    # 2. Check for excluded extensions
    excluded_extensions = [
        ".aux", ".log", ".out", ".toc", ".synctex.gz", ".fls", ".fdb_latexmk",
        ".pdf", ".blg"
    ]
    
    violations = []
    for name in namelist:
        for ext in excluded_extensions:
            if name.endswith(ext):
                violations.append((name, ext))
        if name.startswith("._"):
            violations.append((name, "macOS metadata file"))
            
    if not violations:
        print("[PASS] No compilation log files, auxiliary files, or PDFs found in the ZIP.")
    else:
        print("[FAIL] Violations found in ZIP:")
        for name, reason in violations:
            print(f"  - {name} ({reason})")
            
    # 3. Check relative paths & folders
    sections_count = sum(1 for name in namelist if name.startswith("sections/"))
    appendices_count = sum(1 for name in namelist if name.startswith("appendices/"))
    figures_count = sum(1 for name in namelist if name.startswith("figures/"))
    
    print(f"Directory structure info:")
    print(f"  - files in 'sections/': {sections_count}")
    print(f"  - files in 'appendices/': {appendices_count}")
    print(f"  - files in 'figures/': {figures_count}")
    
    print("\nFull file list inside ZIP:")
    for name in sorted(namelist):
        print(f"  - {name}")

print("\nVerification complete.")
