import os
import zipfile

paper_dir = r"c:\Users\saita\OneDrive\Desktop\AI Everyday\Google Turboquant (Day 1)\kvquant-lab\paper"
zip_path = os.path.join(paper_dir, "arxiv_submission.zip")

# List of files/folders to include
includes = [
    # Main files
    "main.tex",
    "main.bbl",
    "references.bib",
    
    # Style and template files
    "fancyhdr.sty",
    "natbib.sty",
    "iclr2025_conference.sty",
    "iclr2025_conference.bst",
    "iclr2026_conference.sty",
    "iclr2026_conference.bst",
    
    # Directories
    "sections",
    "appendices",
    "figures"
]

# Excluded extensions
excluded_extensions = [
    ".aux", ".log", ".out", ".toc", ".synctex.gz", ".fls", ".fdb_latexmk",
    ".pdf", ".blg"
]

def should_exclude(filename):
    # Exclude temporary Mac OS metadata files (AppleDouble)
    if filename.startswith("._"):
        return True
    # Exclude output PNGs of pages
    if filename.startswith("main_page-") and filename.endswith(".png"):
        return True
    # Exclude files with specific extensions
    for ext in excluded_extensions:
        if filename.endswith(ext):
            return True
    return False

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for item in includes:
        full_path = os.path.join(paper_dir, item)
        if os.path.isdir(full_path):
            for root, dirs, files in os.walk(full_path):
                # Filter out directories starting with . or _ (if any)
                dirs[:] = [d for d in dirs if not d.startswith(".") and not d.startswith("._")]
                for file in files:
                    if should_exclude(file):
                        continue
                    file_full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_full_path, paper_dir)
                    zipf.write(file_full_path, rel_path)
                    print(f"Added: {rel_path}")
        else:
            if os.path.exists(full_path) and not should_exclude(item):
                rel_path = os.path.relpath(full_path, paper_dir)
                zipf.write(full_path, rel_path)
                print(f"Added: {rel_path}")

print(f"\nSuccessfully created {zip_path}")
