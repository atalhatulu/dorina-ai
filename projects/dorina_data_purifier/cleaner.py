"""
Dorina Content Cleaner & Stripper (Enriched with Copyright / Legal disclaimer purge)
"""

import re
from typing import Optional, Dict
import trafilatura

BOILERPLATE_PATTERNS = [
    r"(?i)bu web sitesinde çerez(ler)? kullanılmaktadır.*",
    r"(?i)çerez politikamız(ı|a)? (inceleyin|okuyun|göz atın).*",
    r"(?i)kabul et( ve kapat)?",
    r"(?i)tüm hakları saklıdır.*",
    r"(?i)copyright\s*©?.*",
    r"(?i)abone ol(un)?",
    r"(?i)bizi takip edin.*",
    r"(?i)paylaş:\s*(facebook|twitter|whatsapp|linkedin|x)?",
    r"(?i)yorum(lar)?\s*\(\d+\)",
    r"(?i)ilgili haberler.*",
    r"(?i)son dakika gelişmelerinden haberdar olmak için.*",
    r"(?i)gizlilik sözleşmesi(ni)? okudum.*",
    r"(?i)reklam(lar)?",
    r"(?i)izin alınmaksızın paylaşılamaz.*",
    r"(?i)kaynak gösterilmeksizin (kopyalanamaz|kullanılamaz).*",
    r"(?i)bu sitedeki içerikler.*kopyalanamaz.*",
    r"(?i)bu bölümü yapay zekâ hazırlar.*",
]

COMPILED_BOILERPLATE = [re.compile(p) for p in BOILERPLATE_PATTERNS]

class ContentCleaner:
    def __init__(self, min_word_count: int = 40):
        self.min_word_count = min_word_count

    def clean_html(self, html_content: str, url: Optional[str] = None) -> Optional[Dict]:
        if not html_content or len(html_content.strip()) < 100:
            return None

        extracted_text = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            url=url,
            output_format="txt"
        )

        if not extracted_text:
            return None

        raw_lines = extracted_text.split("\n")
        cleaned_paragraphs = []
        current_para = []

        for line in raw_lines:
            line_strip = line.strip()
            if not line_strip:
                if current_para:
                    cleaned_paragraphs.append(" ".join(current_para))
                    current_para = []
                continue

            is_junk = False
            for pat in COMPILED_BOILERPLATE:
                if pat.search(line_strip):
                    if len(line_strip) < 250:
                        is_junk = True
                        break

            if not is_junk:
                current_para.append(line_strip)

        if current_para:
            cleaned_paragraphs.append(" ".join(current_para))

        final_text = "\n\n".join(cleaned_paragraphs)
        words = final_text.split()
        if len(words) < self.min_word_count:
            return None

        sentences = [s.strip() for s in re.split(r'[.!?]+', final_text) if len(s.strip()) > 3]

        title = ""
        if cleaned_paragraphs:
            first_line = cleaned_paragraphs[0].split("\n")[0]
            title = first_line[:120]

        return {
            "title": title,
            "text": final_text,
            "char_count": len(final_text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len(cleaned_paragraphs)
        }
