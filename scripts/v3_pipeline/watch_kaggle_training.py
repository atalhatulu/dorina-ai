import time
import os
import sys
import kaggle

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

def main():
    log("=" * 60)
    log("Dorina V3 Otonom Kaggle Nobetcisi Devrede!")
    log("=" * 60)

    api = kaggle.KaggleApi()
    api.authenticate()
    kernel_slug = 'beytullahtl/dorina-v3-train-ddp'

    poll_count = 0
    was_running = False

    while True:
        poll_count += 1
        try:
            status_resp = api.kernels_status(kernel_slug)
            # ApiGetKernelSessionStatusResponse nesnesi attribute veya string olarak gelebilir
            if hasattr(status_resp, 'status'):
                status = str(status_resp.status)
            elif isinstance(status_resp, dict):
                status = str(status_resp.get('status', ''))
            else:
                status = str(status_resp)

            # Durum formatını sadeleştir
            status_clean = "RUNNING" if "RUNNING" in status else ("COMPLETE" if "COMPLETE" in status else ("QUEUED" if "QUEUED" in status else ("ERROR" if "ERROR" in status else status)))

            log(f"Denetim #{poll_count} | Durum: {status_clean}")

            if status_clean == "RUNNING":
                if not was_running:
                    log(">>> MUJDE: GPU tahsis edildi! Model egitimi basladi! <<<")
                    was_running = True
                
            elif status_clean == "COMPLETE":
                log("=" * 60)
                log(">>> EGITIM TAMAMLANDI! <<<")
                log("Agirliklar ve kontrol noktalari indiriliyor...")
                os.makedirs("models/dorina_v3_180m/shard3", exist_ok=True)
                os.system(f"kaggle kernels output {kernel_slug} -p models/dorina_v3_180m/shard3")
                log("Agirliklar basariyla indirildi. Otonom nobetci gorevini tamamladi.")
                log("=" * 60)
                break

            elif status_clean == "ERROR":
                log("HATA TESPIT EDILDI!")
                os.system(f"kaggle kernels output {kernel_slug} -p kaggle_kernel/error_logs")
                break

        except Exception as e:
            log(f"API Hatasi (yeniden denenecek): {e}")

        # 30 saniye bekle
        time.sleep(30)

if __name__ == "__main__":
    main()
