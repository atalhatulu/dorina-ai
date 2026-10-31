#!/usr/bin/env python3
"""
Dorina Tokenizer V3: korpus hazırlama, eğitim ve kabul testleri (tek dosya).

  python scripts/train_tokenizer_v3.py stage                  # kotalı korpus -> data/tok_v3/*.jsonl
  python scripts/train_tokenizer_v3.py train                  # tokenizer eğit + manifest yaz
  python scripts/train_tokenizer_v3.py test --baseline dorina_tokenizer_32k.json
  python scripts/train_tokenizer_v3.py all --total-mb 500     # stage + train + test

Notlar
  * Kaynaklar aşağıdaki SOURCES içinde ya da --sources kaynaklar.json ile verilir.
  * Bir kaynak sessizce boş kalmasın diye: eşleşmeyen yol hata verir, kategori kotasının
    %90'ı dolmazsa stage durur (--allow-shortfall ile geçilebilir).
  * GPU'daki eğitimi yavaşlatmamak için varsayılan olarak CPU çekirdeklerinin yarısı
    kullanılır (RAYON_NUM_THREADS ortam değişkeniyle değiştir).
"""
import os, sys, re, json, glob, time, random, hashlib, argparse, unicodedata
from collections import Counter
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import tokenizers
from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, decoders, normalizers

# ───────────────────────── 1. DONDURULACAK KARARLAR ─────────────────────────
VOCAB = 32768
NAMED_SPECIALS = [
    "<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>",
    "<|dorina_thought|>", "<|dorina_thought_end|>",
    "<|tool_call|>", "<|tool_call_end|>", "<|tool_response|>", "<|tool_response_end|>",
    "<|code_start|>", "<|code_end|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
]
SPECIALS = NAMED_SPECIALS + [f"<|reserved{i:03d}|>" for i in range(128)]  # 143 adet, ID 0..142
assert len(SPECIALS) == 143

_CONTR = r"(?i:'(?:s|t|re|ve|m|ll|d))(?!\p{L})"   # İngilizce kısaltma; Türkçe 'de/'da bozulmaz
_WORD = r"[^\r\n\p{L}\p{N}]?\p{L}+"
_DIGIT = r"\p{N}"                                  # tekil rakam
_PUNCT = r" ?[^\s\p{L}\p{N}]+[\r\n]*"
PATTERNS = {
    # doğrulanmış varsayılan: "\n" ve girinti ayrı parçalar
    "split": "|".join([_CONTR, _WORD, _DIGIT, _PUNCT, r"\s*[\r\n]+", r"\s+(?!\S)", r"\s+"]),
    # seçenek: satır sonu + girinti tek parça (kodda satır başına 1 token az)
    "merge": "|".join([_CONTR, _WORD, _DIGIT, _PUNCT, r"\s*[\r\n]+[^\S\r\n]*(?=\S)",
                       r"\s*[\r\n]+", r"\s+(?!\S)", r"\s+"]),
}

MIX = {"english": 0.50, "code": 0.25, "math": 0.10, "turkish": 0.15}

# DİKKAT: yollar ve alan adları senin klasörüne göre düzenlenmeli; HF alan adlarını doğrula.
# Küçük kaynakları öne, büyük kaynağı sona koy: açık kalan kota sonraki kaynağa devreder.
SOURCES = {
    "english": [
        {"type": "hf", "name": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT",
         "split": "train", "fields": ["text"], "weight": 1.0},
    ],
    "code": [
        {"type": "hf", "name": "flytech/python-codes-25k", "split": "train",
         "fields": ["output"], "weight": 0.1},
        {"type": "hf", "name": "codeparrot/codeparrot-clean-valid",
         "split": "train", "fields": ["content"], "weight": 0.9},
    ],
    "math": [
        {"type": "hf", "name": "openai/gsm8k", "config": "main", "split": "train",
         "fields": ["question", "answer"], "weight": 0.1},
        {"type": "hf", "name": "open-web-math/open-web-math", "split": "train",
         "fields": ["text"], "weight": 0.9},
    ],
    "turkish": [
        {"type": "hf", "name": "wikimedia/wikipedia", "config": "20231101.tr",
         "split": "train", "fields": ["text"], "weight": 1.0},
    ],
}

# ───────────────────────── 2. KORPUS HAZIRLAMA ─────────────────────────
_SPECIAL_RE = re.compile("|".join(re.escape(t) for t in SPECIALS))
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RUN = re.compile(r"[ \t]{33,}")


def clean(doc):
    doc = _SPECIAL_RE.sub("", doc)                     # metindeki özel token yazıları silinir
    doc = doc.replace("\r\n", "\n").replace("\r", "\n")
    doc = _CTRL_RE.sub("", doc)
    doc = _WS_RUN.sub(lambda m: m.group(0)[:32], doc)  # uzun boşluk blokları vocab'ı israf eder
    doc = doc.strip()
    if len(doc) < 40:
        return None
    if max(len(l) for l in doc.split("\n")) > 8000:    # minified kod, base64 vb.
        return None
    if not any(c.isalpha() for c in doc[:4000]):
        return None
    return doc


def chunk_text(path, target=64_000):
    buf, size = [], 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf.append(line)
            size += len(line)
            if size >= target:
                yield "".join(buf)
                buf, size = [], 0
    if buf:
        yield "".join(buf)


def iter_files(s):
    pats = s["path"] if isinstance(s["path"], list) else [s["path"]]
    paths = sorted({p for pat in pats for p in glob.glob(pat, recursive=True) if os.path.isfile(p)})
    if not paths:
        raise FileNotFoundError(f"Eşleşen dosya yok: {pats}  (yolu düzenle ya da kaynağı kapat)")
    for p in paths:
        if p.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
            except ImportError:
                raise SystemExit("parquet için `pip install pyarrow` gerekli.")
            field = s.get("parquet_field", "text")
            for batch in pq.ParquetFile(p).iter_batches(batch_size=512, columns=[field]):
                for t in batch.column(0).to_pylist():
                    if t:
                        yield t
            continue
        with open(p, encoding="utf-8", errors="ignore") as f:
            head = f.read(2_000_000)
        for bad in s.get("guard_not_contains", []):
            if bad in head:
                raise SystemExit(f"[{p}] tokenize edilmiş metin izi {bad!r} var; ham sürümü kullan ya da kaynağı çıkar.")
        if p.endswith(".jsonl"):
            field = s.get("jsonl_field", "text")
            with open(p, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        yield json.loads(line)[field]
                    except Exception:
                        continue
        elif s.get("whole_file") and os.path.getsize(p) <= 1_000_000:
            yield open(p, encoding="utf-8", errors="ignore").read()
        else:
            yield from chunk_text(p)


def iter_hf(s):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("HF kaynakları için `pip install datasets` gerekli (ya da --no-hf).")
    ds = load_dataset(s["name"], s.get("config"), split=s.get("split", "train"), streaming=True)
    for row in ds:
        parts = [row[f] for f in s["fields"] if row.get(f)]
        if parts:
            yield "\n\n".join(parts)


def stage(args):
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    total = int(args.total_mb * 1024 * 1024)
    seen, report, failed = set(), {}, []
    for cat, share in MIX.items():
        quota = int(total * share)
        srcs = [s for s in SOURCES[cat] if s.get("enabled", True) and not (s["type"] == "hf" and args.no_hf)]
        if not srcs:
            failed.append(f"{cat}: etkin kaynak yok")
            continue
        remaining, cat_docs = quota, 0
        per_src = []
        with open(work / f"{cat}.jsonl", "w", encoding="utf-8") as ftr, \
             open(work / f"{cat}_heldout.jsonl", "w", encoding="utf-8") as fho:
            for i, s in enumerate(srcs):
                wleft = sum(x["weight"] for x in srcs[i:])
                budget = int(remaining * s["weight"] / wleft)
                got = 0
                it = iter_hf(s) if s["type"] == "hf" else iter_files(s)
                for doc in it:
                    doc = clean(doc)
                    if doc is None:
                        continue
                    h = hashlib.md5(doc.encode("utf-8")).digest()
                    if h in seen:
                        continue
                    seen.add(h)
                    f = fho if int.from_bytes(h[:2], "big") % args.heldout_every == 0 else ftr
                    f.write(json.dumps({"t": doc}, ensure_ascii=False) + "\n")
                    got += len(doc.encode("utf-8"))
                    cat_docs += 1
                    if got >= budget:
                        break
                remaining -= got
                name = s.get("name") or s.get("path")
                per_src.append({"source": name, "bytes": got, "budget": budget})
                print(f"  [{cat}] {name}: {got/1e6:.1f} MB / bütçe {budget/1e6:.1f} MB")
        done = quota - remaining
        report[cat] = {"quota_bytes": quota, "bytes": done, "docs": cat_docs, "sources": per_src}
        print(f"[{cat}] toplam {done/1e6:.1f} / {quota/1e6:.1f} MB")
        if done < 0.9 * quota:
            failed.append(f"{cat}: kotanın %{100*done/quota:.0f}'i doldu")
    (work / "stage_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failed and not args.allow_shortfall:
        raise SystemExit("KOTA EKSİK -> durdum:\n  " + "\n  ".join(failed) +
                         "\nKaynak ekle, --total-mb düşür ya da --allow-shortfall kullan.")


# ───────────────────────── 3. EĞİTİM ─────────────────────────
def make_tokenizer(indent):
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERNS[indent]), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def train(args):
    work = Path(args.work)
    files = [work / f"{c}.jsonl" for c in MIX]
    for f in files:
        if not f.exists():
            raise SystemExit(f"{f} yok; önce `stage` çalıştır.")
    n_docs = sum(1 for f in files for _ in open(f, encoding="utf-8"))

    def docs():
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    yield json.loads(line)["t"]   # newline'lar korunur (girinti merge'leri için şart)

    tok = make_tokenizer(args.indent)
    trainer = trainers.BpeTrainer(vocab_size=args.vocab, min_frequency=2, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    t0 = time.time()
    tok.train_from_iterator(docs(), trainer=trainer, length=n_docs)
    print(f"eğitim {time.time()-t0:.0f} sn, vocab={len(tok.get_vocab())}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    stage_manifest = json.loads((work / "stage_manifest.json").read_text(encoding="utf-8")) \
        if (work / "stage_manifest.json").exists() else {}
    manifest = {
        "tokenizer_file": str(out), "sha256": sha256(out), "vocab_size": len(tok.get_vocab()),
        "indent_mode": args.indent, "pre_tokenizer_regex": PATTERNS[args.indent],
        "normalizer": "NFC", "specials": SPECIALS, "mix": MIX, "corpus": stage_manifest,
        "corpus_files_sha256": {f.name: sha256(f) for f in files},
        "tokenizers_version": tokenizers.__version__, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    mpath = out.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"kaydedildi: {out}\nmanifest : {mpath}")


# ───────────────────────── 4. KABUL TESTLERİ ─────────────────────────
def load_docs(path, limit_bytes):
    docs, n = [], 0
    if not Path(path).exists():
        return docs
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)["t"]
            docs.append(d)
            n += len(d.encode("utf-8"))
            if n >= limit_bytes:
                break
    return docs


def eff(tok, docs):
    if not docs:
        return None
    encs = tok.encode_batch(docs)
    ntok = sum(len(e.ids) for e in encs)
    nbytes = sum(len(d.encode("utf-8")) for d in docs)
    nwords = sum(len(d.split()) for d in docs)
    return {"bytes_per_token": nbytes / ntok, "tokens_per_word": ntok / max(nwords, 1), "tokens": ntok}


def fuzz_strings(n, seed=0):
    rnd = random.Random(seed)
    pools = [
        [chr(c) for c in range(0x20, 0x7F)], ["\n", "\t", " ", "  ", "    "],
        [chr(c) for c in range(0xA0, 0x180)], list("çğıöşüÇĞİÖŞÜ"),
        [chr(c) for c in range(0x370, 0x450)], [chr(c) for c in range(0x4E00, 0x4F00)],
        [chr(c) for c in range(0x1F300, 0x1F400)], [chr(c) for c in range(0x01, 0x20)],
    ]
    out = []
    while len(out) < n:
        s = "".join(rnd.choice(rnd.choice(pools)) for _ in range(rnd.randint(1, 60)))
        if "<|" not in s:
            out.append(s)
    return out


def test(args):
    tok = Tokenizer.from_file(str(args.out))
    res = []

    def add(name, status, detail=""):
        res.append({"test": name, "status": status, "detail": detail})

    def pieces_of(s):
        return [tok.decode([i], skip_special_tokens=False) for i in tok.encode(s).ids]

    # 1) vocab ve ID düzeni
    vocab = tok.get_vocab()
    ids = sorted(vocab.values())
    contiguous = ids == list(range(len(ids)))
    add("vocab_boyutu", "PASS" if len(vocab) == args.vocab and contiguous else "FAIL",
        f"{len(vocab)} token (hedef {args.vocab}), ID'ler ardışık={contiguous}")

    # 2) özel token ID'leri
    bad = [(i, t, tok.token_to_id(t)) for i, t in enumerate(SPECIALS) if tok.token_to_id(t) != i]
    add("ozel_tokenlar", "PASS" if not bad else "FAIL", f"143 beklenen, hatalı: {bad[:5]}")

    # 3) rakamlar tekil
    problems = []
    for s in ["0123456789", "2026", "1000000", "3.14159", "12345 + 67890 = 80235"]:
        dig = [p for p in pieces_of(s) if p.isdigit()]
        if not all(len(p) == 1 for p in dig) or "".join(dig) != "".join(c for c in s if c.isdigit()):
            problems.append((s, pieces_of(s)))
    add("rakamlar_tekil", "PASS" if not problems else "FAIL", str(problems[:2]))

    # 4) Türkçe harfler tek token (küçük harfler + İ zorunlu; Ç Ğ Ö Ş Ü yalnızca BÜYÜK-HARFLİ kelimelerde sık, uyarı)
    bad = [c for c in "çğıöşüİ" if len(tok.encode(c).ids) != 1]
    warn = [c for c in "ÇĞÖŞÜ" if len(tok.encode(c).ids) != 1]
    add("turkce_harfler_tek_token", "FAIL" if bad else ("WARN" if warn else "PASS"),
        f"zorunlu eksik: {bad} | büyük harf uyarı: {warn}")

    # 5) kesme işareti ön-bölmesi
    pp = [p for p, _ in tok.pre_tokenizer.pre_tokenize_str("Türkiye'de İstanbul'da Ankara'dan")]
    pe = [p for p, _ in tok.pre_tokenizer.pre_tokenize_str("I don't know, they'd go, DON'T")]
    ok = all(x in pp for x in ["'de", "'da", "'dan"]) and all(x in pe for x in ["'t", "'d", "'T"])
    add("kesme_isareti", "PASS" if ok else "FAIL", f"TR={pp} EN={pe}")

    # 6) NFC değişmezliği
    bad = [s for s in ["café", "İstanbul", "naïve", "öğrenci", "Ångström"]
           if tok.encode(s).ids != tok.encode(unicodedata.normalize("NFD", s)).ids]
    add("nfc_degismezligi", "PASS" if not bad else "FAIL", f"farklı çıkanlar: {bad}")

    # 7) girinti: kelimeden önceki token sayısı (4/8/12/16 zorunlu; 2 ve 24 yalnızca uyarı)
    limit = 1 if args.indent == "merge" else 2
    rows, bad, soft = [], [], []
    for n in (2, 4, 8, 12, 16, 24):
        toks = tok.encode("\n" + " " * n + "return").tokens
        before = len(toks) - 1
        rows.append((n, before))
        if before > limit or not toks[-1].endswith("return"):
            (bad if n in (4, 8, 12, 16) else soft).append(n)
    add("girinti", "FAIL" if bad else ("WARN" if soft else "PASS"),
        f"mod={args.indent}, izin<={limit}; (boşluk,token)={rows}; zorunlu hata={bad}, uyarı={soft}")

    # 8) roundtrip fuzz
    strs = fuzz_strings(args.fuzz)
    dec = tok.decode_batch([e.ids for e in tok.encode_batch(strs)])
    bad = [(a, b) for a, b in zip(strs, dec) if b != unicodedata.normalize("NFC", a)]
    add("roundtrip_fuzz", "PASS" if not bad else "FAIL", f"{len(strs)} örnek, hata={len(bad)} {bad[:1]}")

    # 9) verimlilik (held-out) ve baseline karşılaştırması
    work = Path(args.work)
    base = Tokenizer.from_file(args.baseline) if args.baseline else None
    sample_docs = {}
    for cat in MIX:
        docs = load_docs(work / f"{cat}_heldout.jsonl", 4_000_000)
        sample_docs[cat] = docs
        e = eff(tok, docs)
        if e is None:
            add(f"verimlilik_{cat}", "WARN", "held-out yok (önce stage; küçük korpusta --heldout-every düşür)")
            continue
        detail = f"{e['bytes_per_token']:.2f} byte/token, {e['tokens_per_word']:.2f} token/kelime"
        status = "INFO"
        if base is not None:
            b = eff(base, docs)
            ratio = e["bytes_per_token"] / b["bytes_per_token"]
            detail += f" | baseline {b['bytes_per_token']:.2f} (oran {ratio:.3f})"
            if cat in ("english", "code"):
                status = "FAIL" if ratio < 0.95 else ("WARN" if ratio < 0.97 else "INFO")
        if cat == "turkish" and e["tokens_per_word"] > 3.0:
            status = "WARN"
        add(f"verimlilik_{cat}", status, detail)

    # 10) vocab sağlığı: kullanılmayan token oranı + en uzun tokenlar
    counts = Counter()
    for cat in MIX:
        for e in tok.encode_batch(load_docs(work / f"{cat}.jsonl", 8_000_000)):
            counts.update(e.ids)
    unused = [i for i in range(len(SPECIALS), len(vocab)) if counts[i] == 0]
    add("kullanilmayan_token_orani", "INFO",
        f"{100*len(unused)/max(1, len(vocab)-len(SPECIALS)):.1f}% (örnekte hiç görülmeyen)")
    longest = sorted(((len(tok.decode([i])), i) for i in range(len(SPECIALS), len(vocab))), reverse=True)[:25]
    add("en_uzun_tokenlar", "INFO", " | ".join(repr(tok.decode([i])) for _, i in longest))

    # 11) hız
    big = [d for docs in sample_docs.values() for d in docs][:2000]
    if big:
        nb = sum(len(d.encode("utf-8")) for d in big)
        t0 = time.time(); tok.encode_batch(big); dt = time.time() - t0
        add("encode_hizi", "INFO", f"{nb/1e6/max(dt,1e-9):.1f} MB/s")

    # rapor
    width = max(len(r["test"]) for r in res)
    for r in res:
        print(f"[{r['status']:4}] {r['test']:<{width}}  {r['detail']}")
    rpath = Path(args.out).with_suffix(".report.json")
    rpath.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    nfail = sum(r["status"] == "FAIL" for r in res)
    print(f"\nrapor: {rpath}  |  FAIL={nfail}  WARN={sum(r['status']=='WARN' for r in res)}")
    if nfail:
        raise SystemExit(1)


# ───────────────────────── 5. CLI ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "train", "test", "all"])
    ap.add_argument("--work", default="data/tok_v3")
    ap.add_argument("--out", default="tokenizers/dorina_tokenizer_v3_32k.json")
    ap.add_argument("--vocab", type=int, default=VOCAB)
    ap.add_argument("--total-mb", type=float, default=500)
    ap.add_argument("--indent", choices=list(PATTERNS), default="split")
    ap.add_argument("--no-hf", action="store_true", help="HF kaynaklarını atla (yalnızca yerel dosyalar)")
    ap.add_argument("--allow-shortfall", action="store_true")
    ap.add_argument("--sources", help="SOURCES'u değiştiren JSON dosyası")
    ap.add_argument("--baseline", help="karşılaştırılacak eski tokenizer.json")
    ap.add_argument("--fuzz", type=int, default=20000)
    ap.add_argument("--heldout-every", type=int, default=100, help="her N. doküman held-out'a ayrılır")
    args = ap.parse_args()
    if args.sources:
        SOURCES.update(json.loads(Path(args.sources).read_text(encoding="utf-8")))
    if args.cmd in ("stage", "all"):
        stage(args)
    if args.cmd in ("train", "all"):
        train(args)
    if args.cmd in ("test", "all"):
        test(args)


if __name__ == "__main__":
    main()
