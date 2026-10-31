"""
Dorina Sitemap & Content Harvester (Freshness & Article-first Sorting)
Sorts URLs newest-first (descending) and skips podcast/gif/media links before fetching.
"""

import xml.etree.ElementTree as ET
import logging
from typing import List
from stealth_fetcher import StealthFetcher

logger = logging.getLogger("SitemapHarvester")

# Doğrudan elenecek URL kalıpları (podcast duyuruları, video gömüleri, medya uzantıları)
MEDIA_FILTER = [
    ".jpg", ".png", ".webp", ".gif", ".jpeg", ".pdf",
    "/tag/", "/etiket/", "/kategori/", "/author/", "/yazar/",
    "/video-", "/podcast-", "/karikatur-", "wp-content/uploads"
]

class SitemapHarvester:
    def __init__(self, fetcher: StealthFetcher):
        self.fetcher = fetcher

    def get_all_sitemap_urls(self, sitemap_url: str, depth: int = 0) -> List[str]:
        if depth > 3:
            return []

        logger.info(f"Sitemap taranıyor (Derinlik {depth}): {sitemap_url}")
        content = self.fetcher.fetch(sitemap_url)
        if not content:
            logger.error(f"Sitemap indirilemedi: {sitemap_url}")
            return []

        urls = []
        sub_sitemaps = []

        try:
            root = ET.fromstring(content)
            tag = root.tag.lower()

            if "sitemapindex" in tag:
                for loc in root.iter():
                    if loc.tag.endswith('loc') and loc.text:
                        sub_sitemaps.append(loc.text.strip())

                # Post içeren alt haritalar
                content_subs = [
                    s for s in sub_sitemaps 
                    if any(k in s.lower() for k in ["post", "content", "article", "yazi", "haber", "makale", "page="])
                    and not any(neg in s.lower() for neg in ["category", "kategori", "tag", "etiket", "author", "yazar"])
                ]

                # En güncel alt haritalardan başla (ters çevir)
                target_list = list(reversed(content_subs)) if content_subs else list(reversed(sub_sitemaps))
                
                for sub in target_list[:4]:
                    sub_urls = self.get_all_sitemap_urls(sub, depth=depth + 1)
                    urls.extend(sub_urls)

            else:
                for loc in root.iter():
                    if loc.tag.endswith('loc') and loc.text:
                        u = loc.text.strip()
                        if not any(m in u.lower() for m in MEDIA_FILTER):
                            urls.append(u)

        except Exception as e:
            logger.error(f"Sitemap parse hatası ({sitemap_url}): {e}")

        # Tekilleştir ve en güncelden eskiye doğru sırala
        clean_unique = list(dict.fromkeys(urls))
        # En yeni yazılar genellikle listenin sonundadır, ters çevirip en güncelden başlatalım
        clean_unique.reverse()
        return clean_unique

if __name__ == "__main__":
    fetcher = StealthFetcher()
    harvester = SitemapHarvester(fetcher)
    u = harvester.get_all_sitemap_urls("https://sarkac.org/sitemap_index.xml")
    print("Sarkaç Temiz Güncel Makale Sayısı:", len(u))
    print("İlk 5 (En Güncel):\n", "\n".join(u[:5]))
