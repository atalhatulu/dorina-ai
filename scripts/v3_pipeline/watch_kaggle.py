import time
import subprocess
import sys

def watch():
    sys.stdout.flush()
    while True:
        try:
            out = subprocess.check_output(["kaggle", "kernels", "status", "beytullahtl/dorina-v3-train-ddp"]).decode("utf-8")
            if "COMPLETE" in out or "ERROR" in out:
                print(f"[EGITIM SONA ERDI]: {out.strip()}")
                sys.stdout.flush()
                break
        except Exception:
            pass
        # 10 dakikada bir (600 saniye) sessizce yokla
        time.sleep(600)

if __name__ == "__main__":
    watch()
