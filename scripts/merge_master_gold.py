import os
import json
import random
import sys
import glob

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def sanitize_and_merge_master():
    teacher_dir = "data/distilled_teachers"
    output_jsonl = os.path.join(teacher_dir, "dorina_gold_sft_master.jsonl")
    output_json = os.path.join(teacher_dir, "dorina_gold_sft_master.json")

    print("=" * 70)
    print("DORINA-AI: BÜYÜK VERİ DENETİMİ, AYIKLAMA & ALTIN MASTER BİRLEŞTİRME")
    print("=" * 70)

    all_files = sorted(glob.glob(os.path.join(teacher_dir, "*.jsonl")))
    teacher_files = [f for f in all_files if not os.path.basename(f).startswith("dorina_gold_sft_master")]

    raw_count = 0
    clean_samples = []
    rejected_reasons = {}
    seen_prompts = set()
    dedup_count = 0

    forbidden_identity_phrases = [
        "ben bir google", "google tarafından geliştirilen", "openai tarafından eğitilmiş",
        "ben chatgpt", "bir dil modeliyim", "as an ai language model"
    ]

    for fpath in teacher_files:
        fname = os.path.basename(fpath)
        file_valid = 0
        file_rejected = 0

        with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
            for line_idx, line in enumerate(fp):
                line_str = line.strip()
                if not line_str:
                    continue
                raw_count += 1
                try:
                    obj = json.loads(line_str)
                    msgs = obj.get("messages", [])
                    if len(msgs) < 2:
                        rejected_reasons["Eksik mesaj çifti (<2)"] = rejected_reasons.get("Eksik mesaj çifti (<2)", 0) + 1
                        file_rejected += 1
                        continue

                    q = msgs[0]["content"].strip()
                    a = msgs[1]["content"].strip()

                    # 1. Çok kısa veya boş kontrolü
                    if len(q) < 5 or len(a) < 5:
                        rejected_reasons["Aşırı kısa veya boş metin"] = rejected_reasons.get("Aşırı kısa veya boş metin", 0) + 1
                        file_rejected += 1
                        continue

                    # 2. Şablon / Placeholder sorusu (örneğin 'Matematiksel problem sorusu...')
                    if "..." in q and len(q) < 50 and any(w in q.lower() for w in ["sorusu", "örnek", "felsefi soru"]):
                        rejected_reasons["Şablon / Placeholder kukla soru"] = rejected_reasons.get("Şablon / Placeholder kukla soru", 0) + 1
                        file_rejected += 1
                        continue

                    # 3. Yabancı AI kimlik sızıntısı
                    lower_a = a.lower()
                    if any(fb in lower_a for fb in forbidden_identity_phrases):
                        rejected_reasons["Yabancı AI kimlik sızıntısı"] = rejected_reasons.get("Yabancı AI kimlik sızıntısı", 0) + 1
                        file_rejected += 1
                        continue

                    # 4. Mükerrer (Deduplication) Kontrolü (küçük harfli normalizasyon)
                    norm_q = " ".join(q.lower().split())
                    if norm_q in seen_prompts:
                        dedup_count += 1
                        file_rejected += 1
                        continue
                    seen_prompts.add(norm_q)

                    clean_samples.append({
                        "messages": [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a}
                        ]
                    })
                    file_valid += 1

                except Exception as e:
                    rejected_reasons[f"JSON Parse Hatası ({type(e).__name__})"] = rejected_reasons.get(f"JSON Parse Hatası ({type(e).__name__})", 0) + 1
                    file_rejected += 1

        print(f"  * {fname:35s}: {file_valid:5,d} temiz / {file_rejected:3,d} ayıklandı")

    print("-" * 70)
    print(f"Toplam Taranan Ham Örnek: {raw_count:,}")
    print(f"Mükerrer (Duplicate) Ayıklanan: {dedup_count:,}")
    for reason, cnt in rejected_reasons.items():
        print(f"  ⚠️ Ayıklanan ({reason}): {cnt}")
    print(f"\n🌟 NİHAİ TERTEMİZ ALTIN MASTER HAVUZ: {len(clean_samples):,} ÖRNEK!")
    print("-" * 70)

    # Karıştır (Shuffle)
    random.seed(42)
    random.shuffle(clean_samples)

    # 1. ChatML JSONL Kaydet
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in clean_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 2. Saf Okunabilir JSON Kaydet
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(clean_samples, f, indent=2, ensure_ascii=False)

    print(f"📁 Master ChatML JSONL : {output_jsonl} ({os.path.getsize(output_jsonl)/(1024*1024):.2f} MB)")
    print(f"📁 Master Saf JSON    : {output_json} ({os.path.getsize(output_json)/(1024*1024):.2f} MB)")
    print("=" * 70)

    # Kapsamlı Kalite & İstatistik Raporu
    run_quality_audit(clean_samples)

def run_quality_audit(samples):
    print("\n🔬 MASTER ALTIN VERİ KALİTE & KATEGORİ DENETİM RAPORU:")
    print("=" * 70)

    categories = {
        "Python & Algoritma / Kodlama": 0,
        "Matematik / Olasılık & Mantık": 0,
        "Fen, Donanım & Bilgisayar Mimarisi": 0,
        "Felsefe, Psikoloji & Derin Türkçe Muhakeme": 0,
        "Genel Asistanlık, Talimat & Dilbilgisi": 0
    }

    total_q_len = 0
    total_a_len = 0
    code_block_count = 0
    step_by_step_count = 0

    for item in samples:
        q = item["messages"][0]["content"]
        a = item["messages"][1]["content"]
        total_q_len += len(q)
        total_a_len += len(a)

        lower_q = q.lower()
        lower_a = a.lower()

        if "```" in a or "def " in a or "class " in a:
            code_block_count += 1
            categories["Python & Algoritma / Kodlama"] += 1
        elif any(w in lower_q for w in ["olasılık", "kombinasyon", "ebob", "ekok", "denklem", "sayısının", "bölümünden kalan", "türev", "integral", "x +", "f(x)"]):
            categories["Matematik / Olasılık & Mantık"] += 1
        elif any(w in lower_q for w in ["ram", "cpu", "bellek", "donanım", "dna", "hücre", "fotosentez", "yerçekimi", "ışık hızı", "big-o", "işletim sistemi"]):
            categories["Fen, Donanım & Bilgisayar Mimarisi"] += 1
        elif any(w in lower_q for w in ["neden", "hissed", "psikoloji", "felsefe", "insan", "zihin", "varoluş", "vicdan", "yalnızlık", "kitap"]):
            categories["Felsefe, Psikoloji & Derin Türkçe Muhakeme"] += 1
        else:
            categories["Genel Asistanlık, Talimat & Dilbilgisi"] += 1

        if any(w in lower_a for w in ["1. adım", "2. adım", "adım 1", "mantık:", "adım adım", "çözüm:"]):
            step_by_step_count += 1

    total = len(samples)
    print(f"📊 Toplam Soru-Cevap Çifti: {total:,}")
    print(f"📝 Ortalama Soru Uzunluğu : {total_q_len / total:.1f} karakter")
    print(f"💡 Ortalama Cevap Uzunluğu: {total_a_len / total:.1f} karakter (~{total_a_len / (total * 4):.0f} token/örnek)")
    print(f"🧩 Çalışır Kod İçeren Örnek: {code_block_count:,} adet (%{code_block_count*100/total:.1f})")
    print(f"🧠 Adım Adım Akıl Yürüten (CoT): {step_by_step_count:,} adet (%{step_by_step_count*100/total:.1f})")
    print("\n📌 Kategori Dağılımı:")
    for cat, cnt in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(cnt * 30 / total)
        print(f"  * {cat:44s}: {cnt:5,d} (%{cnt*100/total:4.1f}) {bar}")
    print("=" * 70)

if __name__ == "__main__":
    sanitize_and_merge_master()
