"""
Dorina Targeted Purifier Runner
Pulls targeted sitemaps, harvests high-signal URLs, and streams them through the purification pipeline.
"""

import sys
import os
import logging

from stealth_fetcher import StealthFetcher
from sitemap_harvester import SitemapHarvester
from pipeline import PurifierPipeline
from curated_sources import CURATED_SITEMAPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TargetedPurifier")

def run_targeted_test(source_key: str = "evrim_agaci_contents", max_articles: int = 5):
    source = CURATED_SITEMAPS.get(source_key)
    if not source:
        logger.error(f"Kaynak bulunamadı: {source_key}")
        return

    logger.info(f"Seçilen Kaynak: {source['name']} ({source['url']})")
    
    pipeline = PurifierPipeline(output_dir="projects/dorina_data_purifier/output")
    harvester = SitemapHarvester(pipeline.fetcher)

    # 1. Sitemap'ten makale URL'lerini çıkar
    urls = harvester.get_urls_from_sitemap(source["url"], limit=max_articles)
    logger.info(f"{len(urls)} adet makale URL'i arıtma boru hattına gönderiliyor...")

    # 2. Pipeline'da arıt
    pipeline.process_urls(urls)

if __name__ == "__main__":
    run_targeted_test("evrim_agaci_contents", max_articles=5)
