"""
Dorina Rejected URL Logger
Logs all rejected URLs with their rejection reason and timestamp.
"""

import os
import json
import time

class RejectedLogger:
    def __init__(self, log_file: str = "projects/dorina_data_purifier/output/rejected_urls.jsonl"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log(self, source: str, url: str, reason: str):
        record = {
            "source": source,
            "url": url,
            "reason": reason,
            "timestamp": time.time()
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def count(self) -> int:
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    return sum(1 for line in f if line.strip())
            except Exception:
                pass
        return 0
