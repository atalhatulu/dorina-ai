import os
import sys

def build_shard4_launcher():
    with open('kaggle_kernel/train_v3_ddp.py', 'r', encoding='utf-8') as f:
        worker_code = f.read()

    launcher_template = '''import os
import sys
import subprocess

WORKER_CODE = """__WORKER_PLACEHOLDER__"""

def main():
    print("=" * 60)
    print("Dorina V3 Standalone Launcher on Kaggle (Shard 4 - Targeted Medicine Run)")
    print("Starting from 60.5 Score SFT Base -> Target: Step 3500 (Curriculum Infusion)")
    print("=" * 60)

    # 1. GPU kontrolü
    n_gpus = 1
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]).decode("utf-8")
        print("[GPU Info]\\n" + gpu_info.strip())
        lines = [l for l in gpu_info.strip().split("\\n") if l.strip()]
        n_gpus = len(lines)
    except Exception as e:
        print("[GPU Info Error]", e)

    print(f"[GPU Algılandı]: {n_gpus} Adet GPU Aktif!")

    # 2. Dataset bul
    print("\\n[Input Directory Scan]")
    found_train = None
    found_val = None
    found_init = None

    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                path = os.path.join(root, f)
                if "v3_shard4_train.bin" in f:
                    found_train = path
                elif "v3_shard4_val.bin" in f:
                    found_val = path
                elif "dorina_v3_sft_base.pt" in f:
                    found_init = path

    if not found_train or not found_val:
        print("HATA: Shard 4 veri dosyaları /kaggle/input altında bulunamadı!")
        sys.exit(1)

    print(f"[Dataset Resolved] Train: {found_train} | Val: {found_val}")
    if found_init:
        print(f"[SFT Base Ağırlığı Bulundu]: {found_init} -> Eğitim 60.5 skorlu SFT modelinden başlayacak!")
    else:
        print("[UYARI]: SFT Base bulunamadı, sıfırdan başlanacak!")

    # 3. train_v3_ddp.py dosyasını /kaggle/working altına oluştur
    os.makedirs("/kaggle/working", exist_ok=True)
    target_script = "/kaggle/working/train_v3_ddp.py"
    with open(target_script, "w", encoding="utf-8") as f:
        f.write(WORKER_CODE)

    print(f"[Worker Script Staged]: {target_script} ({os.path.getsize(target_script)} bytes)")

    # 4. Efektif Batch ve GPU ayarı
    micro_batch = "16"
    grad_accum = "4" if n_gpus > 1 else "8"
    
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={n_gpus}",
        target_script,
        "--data-train", found_train,
        "--data-val", found_val,
        "--out-dir", "/kaggle/working/checkpoints",
        "--shard-name", "shard4_medicine",
        "--micro-batch", micro_batch,
        "--grad-accum", grad_accum,
        "--resume-step", "0",
        "--max-steps", "3500",
        "--keep-last-n-ckpts", "2",
        "--lr", "2e-4",
        "--min-lr", "3e-5",
        "--eval-interval", "250",
        "--save-interval", "500"
    ]

    if found_init:
        cmd.extend(["--init-weights", found_init])

    print("\\n[Çalıştırılıyor] Komut:", " ".join(cmd))
    sys.stdout.flush()

    ret = subprocess.run(cmd, cwd="/kaggle/working")
    print(f"\\n[Sonuç] Eğitim betiği çıkış kodu: {ret.returncode}")
    sys.exit(ret.returncode)

if __name__ == "__main__":
    main()
'''
    escaped_worker = worker_code.replace('\\', '\\\\').replace('\"\"\"', '\\\"\\\"\\\"')
    full_main = launcher_template.replace('__WORKER_PLACEHOLDER__', escaped_worker)
    with open('kaggle_kernel/main.py', 'w', encoding='utf-8') as f:
        f.write(full_main)
    print("SUCCESS: kaggle_kernel/main.py regenerated for Shard 4!")

if __name__ == '__main__':
    build_shard4_launcher()
