import pyarrow.parquet as pq
import fsspec

def preview_sample(name, url, text_col, score_col=None, min_score=None):
    print("\n" + "="*80)
    print(f"[*] KAYNAK: {name}")
    print("="*80)
    fs, path = fsspec.core.url_to_fs(url, headers={"User-Agent": "Mozilla/5.0"})
    with fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        table = pf.read_row_group(0)
        texts = table[text_col].to_pylist()
        scores = table[score_col].to_pylist() if score_col and score_col in table.column_names else None
        
        found = 0
        for i, t in enumerate(texts):
            if not t or len(t.strip()) < 50:
                continue
            if min_score and scores and scores[i] < min_score:
                continue
            
            sc_str = f" [Egitim Skoru: {scores[i]:.2f}]" if scores else ""
            print(f"\n--- ORNEK METIN {found+1}{sc_str} ---")
            # print cleanly
            clean_preview = t[:500].encode("ascii", "replace").decode("ascii")
            print(clean_preview)
            print("...")
            found += 1
            if found >= 1:
                break

try:
    # 1. FineWeb-Edu
    preview_sample(
        "1. FineWeb-Edu (Ingilizce Genel Kultur & Bilim - %35 Pay)",
        "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/parquet/sample-10BT/train/6.parquet",
        "text", "score", 3.0
    )

    # 2. Python Edu Cleaned
    preview_sample(
        "2. Python Edu Cleaned (Kod, Algoritma, Girinti ve Mantik - %25 Pay)",
        "https://huggingface.co/api/datasets/Avelina/python-edu-cleaned/parquet/default/train/0.parquet",
        "text", "score", 3.0
    )

    # 3. FineMath 3+
    preview_sample(
        "3. FineMath 3+ (Matematiksel Nedensellik ve Problem Cozumu - %15 Pay)",
        "https://huggingface.co/api/datasets/HuggingFaceTB/finemath/parquet/finemath-3plus/train/0.parquet",
        "text"
    )

    # 4. Turkish HQ Web
    preview_sample(
        "4. FineWeb2-HQ Turkish (Turkce Dil Bilgisi & Dogal Metin - %25 Pay)",
        "https://huggingface.co/api/datasets/altaidevorg/fineweb2-hq-turkish/parquet/default/train/0.parquet",
        "text"
    )

except Exception as e:
    print("Hata:", e)
