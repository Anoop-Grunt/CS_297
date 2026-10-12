from pathlib import Path
import zipfile
import sys

zip_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cic_dgg_2025.zip")
output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data")

if not zip_path.exists():
    raise FileNotFoundError(f"ZIP file not found: {zip_path}")

if not zipfile.is_zipfile(zip_path):
    raise RuntimeError("The file is not a valid ZIP archive.")

output_dir.mkdir(parents=True, exist_ok=True)

print(f"Extracting {zip_path}...")
with zipfile.ZipFile(zip_path, "r") as archive:
    archive.extractall(output_dir)

files = [p for p in output_dir.rglob("*") if p.is_file()]

print("Extraction complete.")
print("Output directory:", output_dir.resolve())
print(f"Extracted files: {len(files):,}")
