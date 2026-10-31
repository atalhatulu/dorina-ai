"""
Dorina Stealth Fetcher Engine (Persistent Keep-Alive Session)
Reuses the established browser session, cookies, and HTTP/2 keep-alive pipeline.
Drastically cuts connection latency down to < 200ms per request.
"""

import time
import random
import logging
from typing import Optional, Dict
from curl_cffi import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StealthFetcher")

BASE_HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

class StealthFetcher:
    def __init__(self, impersonate: str = "chrome124", min_delay: float = 0.05, max_delay: float = 0.15):
        self.impersonate = impersonate
        self.min_delay = min_delay
        self.max_delay = max_delay
        # Tek bir oturum üzerinden HTTP/2 ve çerezleri korur
        self.session = requests.Session(impersonate=self.impersonate)
        self.session.headers.update(BASE_HEADERS)

    def _sleep_jitter(self):
        if self.min_delay > 0:
            sleep_time = random.uniform(self.min_delay, self.max_delay)
            time.sleep(sleep_time)

    def fetch(self, url: str, max_retries: int = 3, timeout: int = 15) -> Optional[str]:
        for attempt in range(1, max_retries + 1):
            try:
                self._sleep_jitter()
                # Açık oturum üzerinden hızlı GET
                response = self.session.get(url, timeout=timeout)
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code in [403, 429]:
                    wait = attempt * 2 + random.uniform(1.0, 2.0)
                    logger.warning(f"[{response.status_code}] Kısıtlama! ({url}) - {wait:.1f}s bekleniyor (Deneme {attempt}/{max_retries})")
                    time.sleep(wait)
                    # Oturumu tazele
                    self.session = requests.Session(impersonate=self.impersonate)
                    self.session.headers.update(BASE_HEADERS)
                else:
                    logger.warning(f"HTTP {response.status_code} - {url}")
                    return None
            except Exception as e:
                logger.error(f"İstek hatası ({url}): {e}")
                time.sleep(attempt * 1.0)

        return None
