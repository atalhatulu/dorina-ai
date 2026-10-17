import time
import os
import json
import sys

log_file = r"runs\dorina_v0.14_300m\run1\training_log.json"
pid = 10444

def is_process_running(pid):
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid) # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        return False

initial_mtime = os.path.getmtime(log_file) if os.path.exists(log_file) else 0

while True:
    time.sleep(30)
    if os.path.exists(log_file):
        current_mtime = os.path.getmtime(log_file)
        if current_mtime > initial_mtime:
            print(f"NEW_LOG_DETECTED: mtime={current_mtime}", flush=True)
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            history = data.get("history", [])
            if history:
                latest = history[-1]
                print(f"LATEST_ENTRY: {json.dumps(latest, ensure_ascii=False)}", flush=True)
            break

    if not is_process_running(pid):
        print("PROCESS_TERMINATED", flush=True)
        break
