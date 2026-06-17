import urllib.request
import zipfile
import io
import os

urls = [
    "https://github.com/ICLR/Master-Template/raw/master/iclr2025.zip",
    "https://github.com/ICLR/Master-Template/raw/master/iclr2026.zip"
]

dest_dir = r"c:\Users\saita\OneDrive\Desktop\AI Everyday\Google Turboquant (Day 1)\kvquant-lab\paper"

for url in urls:
    print(f"Downloading from {url}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response:
            zip_data = response.read()
        
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            # list files
            print("Files in zip:")
            for name in z.namelist():
                print(" -", name)
                if name.endswith(".sty") or name.endswith(".bst"):
                    # extract
                    filename = os.path.basename(name)
                    if filename:
                        target_path = os.path.join(dest_dir, filename)
                        with open(target_path, "wb") as f:
                            f.write(z.read(name))
                        print(f"Extracted {filename} to {target_path}")
    except Exception as e:
        print(f"Error for {url}: {e}")
