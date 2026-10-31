"""
Dorina Data Purifier - Main Pipeline Runner
Connects Stealth Fetcher, Content Cleaner, Quality Heuristics, and Deduplication.
Outputs ready-to-train gold corpus in JSONL format.
"""

import os
import json
import time
import logging
from typing import List

from stealth_fetcher import StealthFetcher
from cleaner import ContentCleaner
from quality_filters import QualityFilter
from deduplicator import Deduplicator
from sitemap_harvester import SitemapHarvester

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PurifierPipeline")

class PurifierPipeline:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = os.path.join(self.output_dir, "gold_corpus.jsonl")

        self.fetcher = StealthFetcher(min_delay=0.8, max_delay=1.8)
        self.cleaner = ContentCleaner(min_word_count=60)
        self.filter = QualityFilter()
        self.dedup = Deduplicator()
        self.harvester = SitemapHarvester(self.fetcher)

    def process_urls(self, urls: List[str]):
        logger.info(f"Toplu işleme başladı. Toplam URL: {len(urls)}")
        passed_count = 0
        rejected_count = 0

        with open(self.output_file, "a", encoding="utf-8") as out_f:
            for idx, url in enumerate(urls, 1):
                logger.info(f"[{idx}/{len(urls)}] Sayfa getiriliyor: {url}")
                raw_html = self.fetcher.fetch(url)
                if not raw_html:
                    rejected_count += 1
                    continue

                cleaned_doc = self.cleaner.clean_html(raw_html, url=url)
                if not cleaned_doc:
                    logger.warning(f"-> İçerik çıkarılamadı veya çok kısa.")
                    rejected_count += 1
                    continue

                text = cleaned_doc["text"]

                # Tekrar kontrolü
                if self.dedup.is_duplicate(text):
                    logger.warning(f"-> Kopya/Deduplicated içerik elendi.")
                    rejected_count += 1
                    continue

                # Kalite filtresi
                passed, reason, metrics = self.filter.evaluate(text)
                if not passed:
                    logger.warning(f"-> Kalite kriterini geçemedi: {reason}")
                    rejected_count += 1
                    continue

                # Başarıyla geçen altın veri
                doc_record = {
                    "url": url,
                    "text": text,
                    "word_count": cleaned_doc["word_count"],
                    "metrics": metrics,
                    "timestamp": time.time()
                }

                out_f.write(json.dumps(doc_record, ensure_ascii=False) + "\n")
                out_f.flush()
                passed_count += 1
                logger.info(f"==> [ALTIN VERİ KABUL EDİLDİ] ({cleaned_doc['word_count']} kelime)")

        logger.info("="*50)
        logger.info(f"İşlem Tamamlandı! Kabul Edilen: {passed_count} | Elenen: {rejected_count}")
        logger.info(f"Kayıt Yeri: {self.output_file}")

if __name__ == "__main__":
    pipeline = PurifierPipeline()
    test_urls = [
        "https://tr.wikipedia.org/wiki/Yapay_zek%C3%A2",
        "https://tr.wikipedia.org/wiki/Derin_%C3%B6%C4%9Frenme",
        "https://tr.wikipedia.org/wiki/T%C3%BCrkiye"
    ]
    pipeline.process_urls(test_urls)
