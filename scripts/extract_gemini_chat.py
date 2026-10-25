import re
import json
import os
import ast
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def parse_and_clean_raw_gemini(raw_text):
    print("=" * 60)
    print("Dorina-Alchemist: Evrensel Gemini Sohbet Ayıklayıcı")
    print("=" * 60)

    # 1. Markdown kod blokları içindeki JSON'ları bul (```json ... ``` veya ``` ... ```)
    json_blocks = re.findall(r"```(?:json)?\s*(\[\s*\{.*?\}\s*\])\s*```", raw_text, re.DOTALL)
    
    extracted_items = []
    
    # Bloklar halinde ara
    for block in json_blocks:
        try:
            data = json.loads(block)
            if isinstance(data, list):
                extracted_items.extend(data)
        except Exception:
            # json.loads hata verirse regex ile tekil objeleri kurtar
            pass

    # Eğer markdown blokları dışındaysa veya tüm metin ham kopyalandıysa
    if not extracted_items:
        # Regex ile her {"instruction": ..., "response": ...} objesini doğrudan yakala
        item_pattern = re.compile(r'\{\s*"instruction"\s*:\s*"(.*?)"\s*,\s*"response"\s*:\s*"(.*?)"\s*\}', re.DOTALL)
        for m in item_pattern.finditer(raw_text):
            try:
                # JSON string formatına getirip parse et
                obj_str = m.group(0)
                obj = json.loads(obj_str)
                extracted_items.append(obj)
            except Exception:
                continue

    print(f"Toplam Ham Örnek Yakalandı: {len(extracted_items)}")

    # 2. Kalite ve Sözdizimi Denetimi
    gold_samples = []
    syntax_error_count = 0

    for item in extracted_items:
        inst = item.get("instruction", "").strip()
        resp = item.get("response", "").strip()
        
        if not inst or not resp or len(resp) < 30:
            continue
            
        # Eğer cevap içinde python kodu varsa AST ile doğrula
        py_codes = re.findall(r"```python\s*(.*?)\s*```", resp, re.DOTALL)
        is_valid_code = True
        for code in py_codes:
            try:
                ast.parse(code)
            except SyntaxError:
                is_valid_code = False
                syntax_error_count += 1
                break
                
        if not is_valid_code:
            continue

        # ChatML formatına dönüştür
        chatml = {
            "messages": [
                {"role": "user", "content": inst},
                {"role": "assistant", "content": resp}
            ]
        }
        gold_samples.append(chatml)

    print(f"✅ Geçerli Altın Örnek: {len(gold_samples)}")
    if syntax_error_count > 0:
        print(f"⚠️ Hatalı Kod Nedeniyle Elenen: {syntax_error_count}")

    return gold_samples

def extract_text_from_file(file_path):
    if file_path.lower().endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def main():
    import glob
    raw_dir = "data/raw_chats"
    os.makedirs(raw_dir, exist_ok=True)
    output_file = "data/distilled_teachers/gemini_gold_batch1.jsonl"

    files = glob.glob(f"{raw_dir}/*.pdf") + glob.glob(f"{raw_dir}/*.txt")
    if not files and os.path.exists("data/raw_gemini_dump.txt"):
        files = ["data/raw_gemini_dump.txt"]

    if not files:
        print(f"\n[BİLGİ] Lütfen sohbet PDF veya TXT dosyalarını '{raw_dir}' klasörüne at!")
        return

    all_raw_text = ""
    for fpath in files:
        print(f"[Okunuyor]: {fpath}")
        all_raw_text += "\n" + extract_text_from_file(fpath)

    gold = parse_and_clean_raw_gemini(all_raw_text)
    
    if gold:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for item in gold:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n🎯 BAŞARILI! Altın veriler kaydedildi: {output_file}")
        print(f"Toplam eklenen: {len(gold)} adet altın ChatML örneği!")

if __name__ == "__main__":
    main()

