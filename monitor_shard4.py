import time
import subprocess
import os

token_path = r"C:\Users\Talha\test_Key\kaggle.json"
with open(token_path, "r", encoding="utf-8") as f:
    token = f.read().strip()

os.environ["KAGGLE_API_TOKEN"] = token

kernel = "beytullahtl/dorina-v4-train-shard4"
print(f"[*] Dorina-V4 Shard 4 Canli Takip Baslatildi: {kernel}")

last_len = 0
while True:
    try:
        # Check status
        st = subprocess.run(["kaggle", "kernels", "status", kernel], capture_output=True, text=True)
        status_line = [l for l in st.stdout.split("\n") if "KernelWorkerStatus" in l or "status" in l.lower()]
        
        # Check output
        out = subprocess.run(["kaggle", "kernels", "output", kernel], capture_output=True, text=True)
        lines = [l for l in out.stdout.split("\n") if not l.startswith("Warning:")]
        full_text = "\n".join(lines).strip()
        
        if len(full_text) > last_len:
            new_content = full_text[last_len:]
            print(new_content)
            last_len = len(full_text)
        elif status_line:
            print(f"[{time.strftime('%H:%M:%S')}] {status_line[0].strip()}")
            
        if "COMPLETE" in str(status_line) or "ERROR" in str(status_line):
            print("[!] Egitim sonlandi!")
            break
            
    except Exception as e:
        print(f"[!] Hata: {e}")
        
    time.sleep(30)
