import urllib.request
import zipfile
import io
import os

url = "https://github.com/ICLR/Master-Template/raw/master/iclr2026.zip"
dest_dir = r"c:\Users\saita\OneDrive\Desktop\AI Everyday\Google Turboquant (Day 1)\kvquant-lab\scratch"

print("Downloading zip...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    zip_data = response.read()

with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
    for name in z.namelist():
        if "iclr2026_conference.tex" in name and not "__MACOSX" in name and not name.split("/")[-1].startswith("._"):
            target_path = os.path.join(dest_dir, "iclr2026_conference.tex")
            with open(target_path, "wb") as f:
                f.write(z.read(name))
            print(f"Extracted genuine file to {target_path}")
