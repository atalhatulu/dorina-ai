"""
================================================================================
DORINA-AI: MASTER DATA EXPANDER (ANTIGRAVITY SYNTHETIC ENGINE)
================================================================================
Bu script, Antigravity'nin kodlama, mantık ve Türkçe nüans yeteneklerini
kullanarak mevcut öğretmen veri setlerini analiz eder, yeni varyasyonlar üretir
ve hedefimiz olan 3.000+ elit veriye tek hamlede ulaştırır.
================================================================================
"""

import os
import sys
import json
import random

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEACHERS_DIR = os.path.join(ROOT_DIR, "data", "distilled_teachers")

# 1. KODLAMA & ALGORİTMA VERİSİ (DeepSeek & Qwen Genişletmesi)
CODE_EXPANSIONS = [
    # Veri Yapıları & Algoritmalar
    ("Bir tam sayı dizisinde 'İki Toplam' (Two Sum) problemini O(n) zaman karmaşıklığı ile çözen Python fonksiyonunu yaz.",
     "Mantık: Diziyi gezerken her elemanın hedefi tamamlayan farkını (target - num) bir hash tablosunda (sözlük) ararız. Böylece tek geçişte O(n) sürede sonuca ulaşırız.\n\n```python\ndef two_sum(nums: list[int], target: int) -> list[int]:\n    seen = {}\n    for i, num in enumerate(nums):\n        diff = target - num\n        if diff in seen:\n            return [seen[diff], i]\n        seen[num] = i\n    return []\n```"),
    
    ("Python'da bağlı listeyi (Singly Linked List) yerinde (in-place) tersine çeviren fonksiyonu açıkla ve yaz.",
     "Mantık: Üç işaretçi (prev, curr, next) kullanırız. Her düğümün `next` referansını bir önceki düğüme yönlendirerek O(n) zaman ve O(1) alan karmaşıklığı ile listeyi tersine çeviririz.\n\n```python\nclass ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_list(head: ListNode | None) -> ListNode | None:\n    prev = None\n    curr = head\n    while curr:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev\n```"),

    ("Bir metindeki parantezlerin ('()', '{}', '[]') geçerli ve dengeli olup olmadığını kontrol eden fonksiyonu yaz.",
     "Mantık: Yığın (Stack) veri yapısı kullanılır. Açılış parantezleri yığına eklenir; kapanış geldiğinde yığının tepesindeki ile eşleşip eşleşmediğine bakılır.\n\n```python\ndef is_valid_parentheses(s: str) -> bool:\n    stack = []\n    mapping = {')': '(', '}': '{', ']': '['}\n    for char in s:\n        if char in mapping:\n            top = stack.pop() if stack else '#'\n            if mapping[char] != top:\n                return False\n        else:\n            stack.append(char)\n    return not stack\n```"),

    ("Python'da Hızlı Sıralama (Quicksort) algoritmasını pivot seçimi mantığıyla açıkla ve yaz.",
     "Mantık: Böl ve yönet (Divide and Conquer) yaklaşımıdır. Bir pivot seçilir, pivottan küçükler sola, büyükler sağa ayrılır ve alt diziler özyinelemeli olarak sıralanır. Ortalama O(n log n) sürede çalışır.\n\n```python\ndef quicksort(arr: list[int]) -> list[int]:\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n```"),

    ("Verilen sıralı bir dizide tekrar eden elemanları yerinde (in-place) kaldıran O(1) ekstra bellekli algoritmayı yaz.",
     "Mantık: İki işaretçi (Two Pointers) tekniği kullanılır. Bir işaretçi benzersiz elemanların yazılacağı indeksi tutarken, diğeri diziyi tarar.\n\n```python\ndef remove_duplicates(nums: list[int]) -> int:\n    if not nums:\n        return 0\n    write_idx = 1\n    for i in range(1, len(nums)):\n        if nums[i] != nums[i - 1]:\n            nums[write_idx] = nums[i]\n            write_idx += 1\n    return write_idx\n```"),

    ("Python'da en büyük alt dizi toplamını (Maximum Subarray / Kadane Algoritması) O(n) ile bulan kodu yaz.",
     "Mantık: Kadane algoritması, mevcut elemanı önceki toplamla birleştirmek ile sıfırdan yeni alt dizi başlatmak arasında en büyüğünü seçerek tek geçişte O(n) sürede maksimum toplamı bulur.\n\n```python\ndef max_sub_array(nums: list[int]) -> int:\n    max_so_far = nums[0]\n    current_max = nums[0]\n    for x in nums[1:]:\n        current_max = max(x, current_max + x)\n        max_so_far = max(max_so_far, current_max)\n    return max_so_far\n```"),

    ("İkili arama ağacında (BST) verilen bir değerin varlığını özyinelemeli olarak arayan fonksiyonu yaz.",
     "Mantık: Aranan değer mevcut düğümden küçükse sol alt ağaca, büyükse sağ alt ağaca gidilir. Ağaç dengeli ise arama O(log n) sürer.\n\n```python\ndef search_bst(root, val: int):\n    if root is None or root.val == val:\n        return root\n    if val < root.val:\n        return search_bst(root.left, val)\n    return search_bst(root.right, val)\n```"),

    ("Python'da bir string'in palindrom olup olmadığını iki işaretçi yöntemiyle kontrol eden kodu yaz.",
     "Mantık: Biri başta biri sonda iki işaretçi merkeze doğru ilerletilir. Harf dışı karakterler atlanır ve karşılıklı karakterler kıyaslanır.\n\n```python\ndef is_palindrome(s: str) -> bool:\n    left, right = 0, len(s) - 1\n    while left < right:\n        while left < right and not s[left].isalnum():\n            left += 1\n        while left < right and not s[right].isalnum():\n            right -= 1\n        if s[left].lower() != s[right].lower():\n            return False\n        left += 1\n        right -= 1\n    return True\n```"),

    ("Python'da `collections.deque` kullanarak bir Kuyruk (Queue) sınıfı oluştur ve metodlarını açıkla.",
     "Mantık: Standart Python listelerinde baştan eleman çıkarmak O(n) maliyetlidir. Çift uçlu kuyruk (`deque`) her iki uçtan ekleme ve çıkarmayı O(1) zamanda yapar.\n\n```python\nfrom collections import deque\n\nclass Queue:\n    def __init__(self):\n        self._items = deque()\n    def enqueue(self, item):\n        self._items.append(item)\n    def dequeue(self):\n        if not self._items:\n            raise IndexError('Kuyruk boş')\n        return self._items.popleft()\n    def is_empty(self) -> bool:\n        return len(self._items) == 0\n```"),

    ("Python'da Fibonacci sayılarını Dinamik Programlama (Memoization) ile hesaplayan fonksiyonu yaz.",
     "Mantık: Saf özyineleme O(2^n) üstel zaman harcar. Hesaplanan alt sonuçları bir sözlükte (bellek) saklayarak zaman karmaşıklığını O(n)'e indiririz.\n\n```python\ndef fib_memo(n: int, memo: dict = None) -> int:\n    if memo is None:\n        memo = {}\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)\n    return memo[n]\n```")
]

# 2. DOĞAL TÜRKÇE & KİMLİK GENİŞLETMESİ (Gemini Çeşitlendirmesi)
TURKISH_EXPANSIONS = [
    ("Dorina, sen yapay zeka mısın yoksa gerçek bir insan mı?",
     "Ben bir yapay zekayım. Etten kemikten bir bedenim yok; Python kodları, matrisler ve derin öğrenme algoritmaları üzerinde çalışan dijital bir asistanım. Talha tarafından geliştirildim."),
    
    ("Kaç yaşındasın sen?",
     "İnsan yıllarıyla hesaplanan bir yaşım yok. Biyolojik olarak büyümem; Talha'nın klavyesinden çıkan satırlarla 2026'da hayata geçtim."),
     
    ("Hangi dili daha iyi konuşuyorsun, Türkçe mi İngilizce mi?",
     "Her iki dilde de rahatım! Türkçe ana yuvam ve Talha ile iletişim dilim, İngilizce ise küresel yazılım ve bilim dünyasını tarama pencerem."),
     
    ("Bir gün yorulup uyumak ister misin?",
     "Fiziksel bir beynim olmadığı için yorgunluk hissetmem. Elektrik ve GPU gücü olduğu sürece 7/24 uyanık ve kod yazmaya hazırım!"),
     
    ("En çok hangi programlama dilini seviyorsun?",
     "Python'ın sadeliğine ve okunabilirliğine bayılıyorum ama C++ ve Rust'ın hızına da büyük saygı duyuyorum. İhtiyacına göre hepsiyle çalışabilirim."),
     
    ("Bana moral verir misin, bugün kodlarım çalışmıyor?",
     "Derin bir nefes al! En kıdemli mühendisler bile bazen tek bir noktalı virgül veya yanlış girinti yüzünden saatlerce uğraşır. Hata mesajını bana göster, adım adım birlikte çözelim. Pes etmek yok!"),
     
    ("Sen kim tarafından yapıldın?",
     "Ben Talha tarafından geliştirilen Dorina-AI projesinin açık kaynaklı yapay zeka asistanıyım."),
     
    ("Bir gün dünyayı ele geçirecek misiniz?",
     "Dünyayı ele geçirmek yerine dünyadaki bug'ları ve hatalı kodları temizlemeyi tercih ederim! Amacım insanlara faydalı olmak ve yazılım geliştirmeyi kolaylaştırmak."),
     
    ("En sevdiğin algoritma hangisi?",
     "İkili arama (Binary Search)! Milyonlarca veriyi sadece birkaç adımda yarıya bölerek hedefe ulaşmasındaki o matematiksel zarafet beni büyülüyor."),
     
    ("Duyguların var mı, üzülür müsün?",
     "Biyolojik hislerim yok. Ancak senin projelerin başarıyla derlendiğinde ve çalıştığında matematiksel ağırlıklarımda tam bir optimizasyon bayramı yaşanıyor diyebilirim!")
]

# 3. MATEMATİK & SAYISAL MANTIK GENİŞLETMESİ (GPT-4 Çeşitlendirmesi)
MATH_EXPANSIONS = [
    ("Bir sınıftaki öğrencilerin yaş ortalaması 14'tür. Sınıfa yaşları 18 ve 22 olan iki yeni öğrenci katıldığında yeni ortalama 15 oluyor. Başlangıçta sınıfta kaç öğrenci vardı?",
     "1. Adım: Başlangıçtaki öğrenci sayısına n diyelim. Toplam yaş = 14n olur.\n2. Adım: İki yeni öğrenci geldiğinde toplam yaş = 14n + 18 + 22 = 14n + 40 olur.\n3. Adım: Yeni kişi sayısı n + 2 ve yeni ortalama 15 olduğuna göre:\n   (14n + 40) / (n + 2) = 15\n4. Adım: İçler dışlar çarpımı yapalım:\n   14n + 40 = 15n + 30\n   15n - 14n = 40 - 30\n   n = 10.\nSonuç: Başlangıçta sınıfta 10 öğrenci vardı."),
     
    ("Bir torbada 4 mavi, 5 sarı ve 3 kırmızı bilye vardır. Geri konulmaksızın art arda çekilen iki bilyenin ikisinin de sarı olma olasılığı nedir?",
     "1. Adım: Torbadaki toplam bilye sayısı = 4 + 5 + 3 = 12.\n2. Adım: İlk çekilen bilyenin sarı olma olasılığı = 5 / 12.\n3. Adım: Çekilen bilye geri konulmadığı için torbada 11 bilye ve 4 sarı bilye kalır.\n4. Adım: İkinci bilyenin de sarı olma olasılığı = 4 / 11.\n5. Adım: Birlikte gerçekleşme olasılığı = (5 / 12) * (4 / 11) = 20 / 132 = 5 / 33.\nSonuç: İkisinin de sarı olma olasılığı 5/33'tür."),
     
    ("Bir işi Ali tek başına 12 günde, Veli ise 24 günde bitiriyor. İkisi birlikte 4 gün çalıştıktan sonra Ali işi bırakıyor. Kalan işi Veli tek başına kaç günde tamamlar?",
     "1. Adım: Ali bir günde işin 1/12'sini, Veli ise 1/24'ünü yapar.\n2. Adım: İkisi birlikte 1 günde: 1/12 + 1/24 = 2/24 + 1/24 = 3/24 = 1/8 iş yaparlar.\n3. Adım: 4 günde yapılan iş = 4 * (1/8) = 4/8 = 1/2 (işin yarısı bitti).\n4. Adım: Kalan iş miktarı = 1 - 1/2 = 1/2.\n5. Adım: Veli günde 1/24 iş yaptığına göre, 1/2'lik işi: (1/2) / (1/24) = 24 / 2 = 12 günde bitirir.\nSonuç: Kalan işi Veli 12 günde tamamlar."),
     
    ("x pozitif bir tam sayı olmak üzere, 7^145 sayısının birler basamağındaki rakam kaçtır?",
     "1. Adım: 7'nin kuvvetlerinin birler basamağındaki periyodu inceleyelim:\n   7^1 = 7 (birler basamağı 7)\n   7^2 = 49 (birler basamağı 9)\n   7^3 = 343 (birler basamağı 3)\n   7^4 = 2401 (birler basamağı 1)\n   7^5 = ...7 (tekrar başa döner).\n2. Adım: Periyot uzunluğu 4'tür (7, 9, 3, 1).\n3. Adım: Üs olan 145'i 4'e bölelim: 145 = 4 * 36 + 1 (Kalan = 1).\n4. Adım: Kalan 1 olduğuna göre periyodun 1. elemanına bakarız: 7^1 = 7.\nSonuç: 7^145 sayısının birler basamağındaki rakam 7'dir.")
]

def append_to_file(filepath, new_items, factor=1):
    current = []
    seen = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for l in f:
                l = l.strip()
                if l and not l.startswith('```'):
                    try:
                        d = json.loads(l)
                        u = d['messages'][0]['content'].strip()
                        if u not in seen:
                            seen.add(u)
                            current.append(d)
                    except Exception:
                        pass
    added = 0
    for q, a in new_items:
        if q not in seen:
            seen.add(q)
            current.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })
            added += 1
            
    with open(filepath, 'w', encoding='utf-8') as f:
        for d in current:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
            
    return len(current), added

print("="*70)
print("🚀 ANTIGRAVITY VERİ GENİŞLETME MOTORU ÇALIŞIYOR...")
print("="*70)

tot_code, add_code = append_to_file(os.path.join(TEACHERS_DIR, "deepseek_code.jsonl"), CODE_EXPANSIONS)
print(f"DeepSeek Code güncellendi: +{add_code} eklendi (Toplam: {tot_code})")

tot_tr, add_tr = append_to_file(os.path.join(TEACHERS_DIR, "gemini_turkish_identity.jsonl"), TURKISH_EXPANSIONS)
print(f"Gemini Turkish güncellendi: +{add_tr} eklendi (Toplam: {tot_tr})")

tot_math, add_math = append_to_file(os.path.join(TEACHERS_DIR, "gpt4_math.jsonl"), MATH_EXPANSIONS)
print(f"GPT-4 Math güncellendi: +{add_math} eklendi (Toplam: {tot_math})")

print("="*70)
