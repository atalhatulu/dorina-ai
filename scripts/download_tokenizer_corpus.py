import os
import io
import urllib.request
import pyarrow.parquet as pq

os.makedirs("data/nano_v2_raw", exist_ok=True)
SAMPLE_FILE = "data/nano_v2_raw/tokenizer_corpus.txt"

print(">>> Downloading representative sample corpus for 8k Tokenizer...")

urls = [
    # 1. FineWeb-Edu (40%)
    ("FineWeb-Edu", "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/parquet/sample-10BT/train/0.parquet", 3000),
    # 2. Cosmopedia (25%)
    ("Cosmopedia", "https://huggingface.co/api/datasets/HuggingFaceTB/cosmopedia-100k/parquet/default/train/0.parquet", 2500),
    # 3. Simple Wiki (15%)
    ("SimpleWiki", "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.simple/train/0.parquet", 2000),
]

with open(SAMPLE_FILE, "w", encoding="utf-8") as out_f:
    for name, url, num_rows in urls:
        print(f"Fetching sample from {name}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        # Read first ~25MB of parquet
        with urllib.request.urlopen(req) as resp:
            data = resp.read(25 * 1024 * 1024)
        
        reader = pq.ParquetFile(io.BytesIO(data))
        table = reader.read_row_group(0)
        df = table.to_pandas()
        
        col = "text" if "text" in df.columns else df.columns[0]
        count = 0
        for text in df[col].dropna():
            t = str(text).strip()
            if len(t) > 100:
                out_f.write(t + "\n\n")
                count += 1
                if count >= num_rows:
                    break
        print(f"Collected {count} documents from {name}.")

size_mb = os.path.getsize(SAMPLE_FILE) / (1024 * 1024)
print(f"Successfully created tokenizer sample corpus: {SAMPLE_FILE} ({size_mb:.2f} MB)")
