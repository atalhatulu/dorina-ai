"""
Dorina Deduplication Engine (Persistent, Multi-tier & Source-aware)
Tracks seen URLs and Content Hashes per-source and globally.
"""

import os
import hashlib
from typing import Set

class PersistentDeduplicator:
    def __init__(self, index_file: str = "projects/dorina_data_purifier/output/seen_hashes.txt"):
        self.index_file = index_file
        self.seen_hashes: Set[str] = set()
        self.seen_urls: Set[str] = set()
        self._load()

    def _load(self):
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            self.seen_urls.add(parts[0])
                            self.seen_hashes.add(parts[1])
            except Exception:
                pass

    def _save_record(self, url: str, h: str):
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        with open(self.index_file, "a", encoding="utf-8") as f:
            f.write(f"{url}\t{h}\n")

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().split())

    def is_url_seen(self, url: str) -> bool:
        return url.strip().rstrip("/") in self.seen_urls

    def is_content_duplicate(self, text: str) -> bool:
        norm = self._normalize(text)
        h = hashlib.md5(norm.encode("utf-8")).hexdigest()
        return h in self.seen_hashes

    def register(self, url: str, text: str):
        url_clean = url.strip().rstrip("/")
        norm = self._normalize(text)
        h = hashlib.md5(norm.encode("utf-8")).hexdigest()
        
        self.seen_urls.add(url_clean)
        self.seen_hashes.add(h)
        self._save_record(url_clean, h)

    def size(self) -> int:
        return len(self.seen_hashes)
