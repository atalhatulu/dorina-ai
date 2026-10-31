import json
import random

# Target rules:
# 1. mi / mı soru eki (geldinmi -> geldin mi, bu dogrumu -> bu doğru mu)
# 2. de / da bağlacı (bende gelicem -> ben de geleceğim, sende gel -> sen de gel)
# 3. Buyuk harf ve kesme isareti (turkiyenin -> Türkiye'nin, ankaraya -> Ankara'ya)
# 4. Sokak dili -> Yazi dili (gelicem -> geleceğim, yapcam -> yapacağım, gidiyom -> gidiyorum)
# 5. Eski gorevler (ASCII & Typo) regresyon onleyici

MI_WORDS = [
    ("geldinmi", "geldin mi"), ("olurmu", "olur mu"), ("gördünmü", "gördün mü"),
    ("bakar mısın", "bakar mısın"), ("anladınmı", "anladın mı"), ("biliyormusun", "biliyor musun"),
    ("gelirmisin", "gelir misin"), ("duydunmu", "duydun mu"), ("doğrumu", "doğru mu"),
    ("dogrumu", "doğru mu"), ("varmı", "var mı"), ("varmi", "var mı"), ("yokmu", "yok mu")
]

DEDA_PAIRS = [
    ("bende gelicem", "ben de geleceğim"),
    ("sende gel", "sen de gel"),
    ("bizde gelelim", "biz de gelelim"),
    ("onlarda biliyor", "onlar da biliyor"),
    ("sen de geliyormusun", "sen de geliyor musun"),
    ("sende haklısın", "sen de haklısın"),
    ("evde kimse yok", "evde kimse yok"), # Bulunma hali - degismez
    ("arabada bekliyorum", "arabada bekliyorum") # Bulunma hali - degismez
]

PROPER_NOUNS = [
    ("turkiyenin baskenti ankaradir", "Türkiye'nin başkenti Ankara'dır."),
    ("istanbulda hava guzel", "İstanbul'da hava güzel."),
    ("ahmetin arabasi bozuldu", "Ahmet'in arabası bozuldu."),
    ("mehmetle bulustuk", "Mehmet'le buluştuk."),
    ("izmire gidiyoruz", "İzmir'e gidiyoruz."),
    ("turkiye buyuk millet meclisi", "Türkiye Büyük Millet Meclisi")
]

SLANG_TO_CLEAN = [
    ("yarin gelicem", "yarın geleceğim"),
    ("ne yapcam bilmiyorum", "ne yapacağım bilmiyorum"),
    ("eve gidiyom simdi", "eve gidiyorum şimdi"),
    ("seni seviyom", "seni seviyorum"),
    ("geliyom bekleyin", "geliyorum bekleyin"),
    ("napıyorsun", "ne yapıyorsun"),
    ("napıyon", "ne yapıyorsun"),
    ("bi dakka geliyom", "bir dakika geliyorum")
]

REGRESSION_ASCII = [
    ("yarin gorusuruz insallah", "yarın görüşürüz inşallah"),
    ("cok tesekkur ederim", "çok teşekkür ederim"),
    ("aksam gorusmek uzere", "akşam görüşmek üzere"),
    ("iyi aksamlar herkese", "iyi akşamlar herkese"),
    ("nasil gidiyor hayat", "nasıl gidiyor hayat"),
    ("Ben ev gidiyrum", "Ben eve gidiyorum")
]

samples = []

# Generate 30,000 synthetic sentences mixing these
templates = [
    "{prefix} {corrupt} {suffix}",
    "{corrupt} {suffix}",
    "{prefix} {corrupt}",
    "{corrupt}"
]

prefixes = ["gerçekten", "bence", "acaba", "yarın", "bugün", "lütfen", "arkadaşlar", "evet", "hayır"]
suffixes = ["acaba?", "diyorum.", "bence de öyle.", "fark ettin mi?", "baksana.", "tamam mı?"]

pairs_all = []

# Collect all base pairs
for c, clean in MI_WORDS:
    pairs_all.append((f"sen {c} dün", f"sen {clean} dün"))
    pairs_all.append((f"{c} sence", f"{clean} sence"))
    pairs_all.append((c, clean))

pairs_all.extend(DEDA_PAIRS)
pairs_all.extend(PROPER_NOUNS)
pairs_all.extend(SLANG_TO_CLEAN)
pairs_all.extend(REGRESSION_ASCII)

print(f"Base high-precision pair count: {len(pairs_all)}")

# Multiply and create diverse dataset
augmented = []
for _ in range(350): # 350x multiplier ~ 25k samples
    for corrupt, clean in pairs_all:
        augmented.append(f"<fix> {corrupt} => {clean}\n")

random.shuffle(augmented)

out_file = "data/nano_tr_data/synthetic_refinement_mix.txt"
with open(out_file, "w", encoding="utf-8") as f:
    f.writelines(augmented)

print(f"Generated {len(augmented):,} synthetic refinement examples in {out_file}")
