import os
import sys
import re
import json
import glob
import ast
import pypdf

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def extract_from_pdf(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text

def parse_pdf_text(raw_text):
    # JSON bloklarını bul (```json ... ``` veya ham liste)
    blocks = re.findall(r"```(?:json)?\s*(\[\s*\{.*?\}\s*\])\s*```", raw_text, re.DOTALL)
    items = []
    
    for b in blocks:
        try:
            data = json.loads(b)
            if isinstance(data, list):
                items.extend(data)
        except Exception:
            pass

    # Regex ile doğrudan instruction/response objelerini yakala
    if not items:
        pattern = re.compile(r'\{\s*"instruction"\s*:\s*"(.*?)"\s*,\s*"response"\s*:\s*"(.*?)"\s*\}', re.DOTALL)
        for m in pattern.finditer(raw_text):
            try:
                obj = json.loads(m.group(0))
                items.append(obj)
            except Exception:
                pass
                
    return items

def main():
    raw_dir = "data/raw_chats"
    output_jsonl = "data/distilled_teachers/gemini_gold_batch1.jsonl"
    output_pure_json = "data/distilled_teachers/gemini_gold_batch1.json"
    
    pdf_files = sorted(glob.glob(f"{raw_dir}/*.pdf"))
    print("=" * 70)
    print(f"DORINA-ALCHEMIST: {len(pdf_files)} ADET PDF İŞLENİYOR")
    print("=" * 70)
    
    all_raw_items = []
    for f in pdf_files:
        print(f"[Okunuyor]: {os.path.basename(f)}...", end=" ")
        try:
            txt = extract_from_pdf(f)
            extracted = parse_pdf_text(txt)
            print(f"-> {len(extracted)} örnek bulundu")
            all_raw_items.extend(extracted)
        except Exception as e:
            print(f"-> Hata: {e}")

    print(f"\nToplam Çekilen Ham Örnek Sayısı: {len(all_raw_items):,}")
    
    # Benzerleri / Duplikeleri Ayıklama (Deduplication)
    seen_instructions = set()
    unique_items = []
    syntax_error_count = 0
    dummy_count = 0

    for item in all_raw_items:
        inst = item.get("instruction", "").strip()
        resp = item.get("response", "").strip()

        # Prompt şablonundaki sahte örneği atla
        if "Kullanıcı sorusu veya çözülmesi istenen problem" in inst or "Gerçek dünya sistem veya veritabanı problemi" in inst:
            dummy_count += 1
            continue

        if len(inst) < 10 or len(resp) < 20:
            continue

        # Benzer soru ayıklama (Normalizasyon ile deduplication)
        norm_key = re.sub(r"\W+", "", inst.lower())
        if norm_key in seen_instructions:
            continue
        seen_instructions.add(norm_key)

        # Python kodu varsa sözdizimi doğrula
        py_codes = re.findall(r"```python\s*(.*?)\s*```", resp, re.DOTALL)
        has_error = False
        for c in py_codes:
            try:
                ast.parse(c)
            except SyntaxError:
                has_error = True
                syntax_error_count += 1
                break

        if not has_error:
            unique_items.append({"instruction": inst, "response": resp})

    print("-" * 70)
    print(f"✅ Tekil & Hatasız Altın Örnek: {len(unique_items):,}")
    print(f"🗑️ Elenen Duplikeler (Aynı Sorular): {len(all_raw_items) - len(unique_items) - dummy_count}")
    print(f"⚠️ Hatalı Kod Nedeniyle Elenen: {syntax_error_count}")
    print("-" * 70)

    # 1. Saf .json listesi olarak kaydet (İstediğin format)
    with open(output_pure_json, "w", encoding="utf-8") as f:
        json.dump(unique_items, f, indent=2, ensure_ascii=False)
    print(f"💾 Saf .json dosyası kaydedildi: {output_pure_json}")

    # 2. Dorina ChatML .jsonl olarak kaydet
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for it in unique_items:
            line = {
                "messages": [
                    {"role": "user", "content": it["instruction"]},
                    {"role": "assistant", "content": it["response"]}
                ]
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"💾 Dorina ChatML .jsonl kaydedildi: {output_jsonl}")
    print("=" * 70)

if __name__ == "__main__":
    main()
