"""
Dorina Data Purifier - Live Studio Web Server
FastAPI + WebSocket server broadcasting real-time scraping & stats at high-speed throughput.
Tracks:
- Sitedeki Toplam
- Toplam Taranan
- Geçerli (Altın)
- Elenen (Çöp/Kısa)
Saves rejected URLs to output/rejected_urls.jsonl with failure reasons.
"""

import os
import json
import asyncio
import logging
from typing import List, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from stealth_fetcher import StealthFetcher
from cleaner import ContentCleaner
from quality_filters import QualityFilter
from deduplicator import PersistentDeduplicator
from sitemap_harvester import SitemapHarvester
from curated_sources import CURATED_SITEMAPS
from rejected_logger import RejectedLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PurifierServer")

app = FastAPI(title="Dorina Data Purifier Studio")

active_connections: Set[WebSocket] = set()

class EngineState:
    def __init__(self):
        self.is_running = False
        self.should_stop = False
        self.current_source = "evrim_agaci_contents"
        self.output_dir = "projects/dorina_data_purifier/output"
        self.sources_dir = os.path.join(self.output_dir, "sources")
        os.makedirs(self.sources_dir, exist_ok=True)
        
        self.dedup = PersistentDeduplicator(index_file=os.path.join(self.output_dir, "seen_hashes.txt"))
        self.rejected_logger = RejectedLogger(log_file=os.path.join(self.output_dir, "rejected_urls.jsonl"))
        self.site_total_articles = 0

    def get_source_file(self, source_key: str) -> str:
        clean_name = "".join(c if c.isalnum() or c == "_" else "_" for c in source_key)
        return os.path.join(self.sources_dir, f"{clean_name}.jsonl")

    def get_source_stats(self, source_key: str):
        filepath = self.get_source_file(source_key)
        articles = 0
        words = 0
        sentences = 0
        chars = 0
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            doc = json.loads(line)
                            articles += 1
                            words += doc.get("word_count", 0)
                            sentences += doc.get("sentence_count", 0)
                            chars += doc.get("char_count", 0)
            except Exception:
                pass
        
        tokens = int(words * 1.35)
        size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 2) if os.path.exists(filepath) else 0.0
        rejected = self.rejected_logger.count()
        scanned = articles + rejected

        return {
            "articles": articles,
            "scanned": scanned,
            "rejected": rejected,
            "words": words,
            "tokens": tokens,
            "sentences": sentences,
            "chars": chars,
            "size_mb": size_mb
        }

state = EngineState()

async def broadcast(event: dict):
    if not active_connections:
        return
    dead = set()
    msg = json.dumps(event)
    for ws in list(active_connections):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    active_connections.difference_update(dead)

def get_stats_payload(source_key: str = None):
    if not source_key:
        source_key = state.current_source
    
    src_stats = state.get_source_stats(source_key)
    remaining = max(0, state.site_total_articles - src_stats["articles"]) if state.site_total_articles > 0 else 0

    return {
        "type": "stats_update",
        "current_source": source_key,
        "total_articles": src_stats["articles"],
        "total_scanned": src_stats["scanned"],
        "total_rejected": src_stats["rejected"],
        "site_total": state.site_total_articles,
        "remaining": remaining,
        "total_words": src_stats["words"],
        "total_tokens": src_stats["tokens"],
        "disk_mb": src_stats["size_mb"],
        "total_sentences": src_stats["sentences"],
        "total_chars": src_stats["chars"]
    }

@app.get("/")
async def get_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)

    await websocket.send_text(json.dumps(get_stats_payload()))
    await websocket.send_text(json.dumps({
        "type": "status",
        "running": state.is_running
    }))

    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)
            action = cmd.get("action")

            if action == "inspect_source":
                source = cmd.get("source")
                state.current_source = source
                asyncio.create_task(inspect_source_task(source))

            elif action == "start":
                if not state.is_running:
                    source = cmd.get("source")
                    custom_url = cmd.get("custom_url")
                    limit = cmd.get("limit", 50)
                    state.current_source = source
                    asyncio.create_task(run_crawler_task(source, custom_url, limit))

            elif action == "stop":
                state.should_stop = True
                await broadcast({
                    "type": "step",
                    "step_index": 0,
                    "message": "Durdurma sinyali gönderildi, mevcut sayfa bittikten sonra duracak...",
                    "level": "warn"
                })

    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def inspect_source_task(source_key: str):
    sitemap_info = CURATED_SITEMAPS.get(source_key, {})
    sitemap_url = sitemap_info.get("url")
    if not sitemap_url:
        return

    fetcher = StealthFetcher(min_delay=0.05, max_delay=0.1)
    harvester = SitemapHarvester(fetcher)
    all_urls = await asyncio.to_thread(harvester.get_all_sitemap_urls, sitemap_url)
    if all_urls:
        state.site_total_articles = len(all_urls)
        await broadcast(get_stats_payload(source_key))

async def run_crawler_task(source_key: str, custom_url: str, limit: int):
    state.is_running = True
    state.should_stop = False
    await broadcast({"type": "status", "running": True})

    try:
        fetcher = StealthFetcher(min_delay=0.05, max_delay=0.15)
        cleaner = ContentCleaner(min_word_count=40)
        qfilter = QualityFilter()
        harvester = SitemapHarvester(fetcher)

        source_output_file = state.get_source_file(source_key)

        await broadcast({
            "type": "step",
            "step_index": 1,
            "message": f"AŞAMA 1: [{source_key}] Sitemap haritası inceleniyor...",
            "level": "info"
        })

        if source_key == "custom" and custom_url:
            sitemap_url = custom_url
        else:
            sitemap_info = CURATED_SITEMAPS.get(source_key, {})
            sitemap_url = sitemap_info.get("url")

        if not sitemap_url:
            await broadcast({
                "type": "step",
                "step_index": 1,
                "message": "Geçerli bir sitemap URL bulunamadı!",
                "level": "error"
            })
            return

        all_site_urls = await asyncio.to_thread(harvester.get_all_sitemap_urls, sitemap_url)
        state.site_total_articles = len(all_site_urls)

        # URL Deduplication
        uncollected_urls = [u for u in all_site_urls if not state.dedup.is_url_seen(u)]
        
        if limit > 0:
            urls_to_crawl = uncollected_urls[:limit]
        else:
            urls_to_crawl = uncollected_urls

        await broadcast({
            "type": "step",
            "step_index": 1,
            "message": f"Sitede Toplam: {len(all_site_urls)} | Kalan: {len(uncollected_urls)}. Şimdi {len(urls_to_crawl)} makale taranacak.",
            "level": "success"
        })
        await broadcast(get_stats_payload(source_key))

        if not urls_to_crawl:
            await broadcast({
                "type": "step",
                "step_index": 1,
                "message": f"[{source_key}] Bu kaynaktaki tüm makaleler zaten toplanmış!",
                "level": "warn"
            })
            return

        with open(source_output_file, "a", encoding="utf-8") as out_f:
            for idx, url in enumerate(urls_to_crawl, 1):
                if state.should_stop:
                    break

                raw_html = await asyncio.to_thread(fetcher.fetch, url)
                if not raw_html:
                    state.rejected_logger.log(source_key, url, "SAYFA_CEKILEMEDI")
                    await broadcast({
                        "type": "doc_rejected",
                        "url": url,
                        "reason": "SAYFA_CEKILEMEDI"
                    })
                    await broadcast(get_stats_payload(source_key))
                    continue

                doc = cleaner.clean_html(raw_html, url=url)
                if not doc:
                    state.rejected_logger.log(source_key, url, "YETERSIZ_ICERIK_VEYA_KISA")
                    await broadcast({
                        "type": "doc_rejected",
                        "url": url,
                        "reason": "YETERSIZ_ICERIK_VEYA_KISA"
                    })
                    await broadcast(get_stats_payload(source_key))
                    continue

                if state.dedup.is_content_duplicate(doc["text"]):
                    state.rejected_logger.log(source_key, url, "KOPYA_ICERIK_DEDUPLICATED")
                    await broadcast({
                        "type": "doc_rejected",
                        "url": url,
                        "reason": "KOPYA_ICERIK_DEDUPLICATED"
                    })
                    await broadcast(get_stats_payload(source_key))
                    continue

                passed, reason, metrics = qfilter.evaluate(doc["text"])
                if not passed:
                    state.rejected_logger.log(source_key, url, reason)
                    await broadcast({
                        "type": "doc_rejected",
                        "url": url,
                        "reason": reason
                    })
                    await broadcast(get_stats_payload(source_key))
                    continue

                # KABUL EDİLEN ALTIN VERİ
                state.dedup.register(url, doc["text"])

                doc_record = {
                    "source": source_key,
                    "url": url,
                    "title": doc["title"],
                    "text": doc["text"],
                    "word_count": doc["word_count"],
                    "char_count": doc["char_count"],
                    "sentence_count": doc["sentence_count"],
                    "paragraph_count": doc["paragraph_count"],
                    "metrics": metrics
                }
                out_f.write(json.dumps(doc_record, ensure_ascii=False) + "\n")
                out_f.flush()

                await broadcast({
                    "type": "doc_accepted",
                    "url": url,
                    "title": doc["title"],
                    "text": doc["text"][:1200] + ("..." if len(doc["text"]) > 1200 else ""),
                    "word_count": doc["word_count"],
                    "char_count": doc["char_count"],
                    "sentence_count": doc["sentence_count"],
                    "paragraph_count": doc["paragraph_count"]
                })

                await broadcast(get_stats_payload(source_key))

    except Exception as e:
        logger.error(f"Genel görev hatası: {e}", exc_info=True)
        await broadcast({
            "type": "step",
            "step_index": 0,
            "message": f"Hata oluştu: {e}",
            "level": "error"
        })
    finally:
        state.is_running = False
        await broadcast({"type": "status", "running": False})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8989)
