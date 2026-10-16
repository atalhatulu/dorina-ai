import os
import sys
import json
import time
import math
import shutil
import glob
import subprocess
from datetime import datetime
import torch
import pynvml

class LabLogger:
    def __init__(self, model_name: str, config: dict, total_steps: int, run_tag: str = None):
        self.model_family = model_name
        self.config = config
        self.total_steps = total_steps
        self.start_time = time.time()
        self.step_start_time = time.time()
        
        self.has_nvml = False
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.has_nvml = True
        except Exception:
            self.has_nvml = False

        self.base_runs_dir = os.path.join("runs", self.model_family)
        os.makedirs(self.base_runs_dir, exist_ok=True)

        is_scratch = config.get("mode") == "scratch"
        existing_runs = sorted(
            glob.glob(os.path.join(self.base_runs_dir, "run*")),
            key=lambda p: int(os.path.basename(p).replace("run", "")) if os.path.basename(p).replace("run", "").isdigit() else 0
        )

        if is_scratch and existing_runs:
            print("\n" + "!" * 70)
            print(f"📦 '{self.model_family}' için önceki eğitim kayıtları tespit edildi.")
            archive_choice = input("Önceki tüm koşular '_archive' dizinine yedeklensin mi? (E/h) [Varsayılan: E]: ").strip().lower()
            if archive_choice != "h":
                archive_timestamp = None
                last_log = os.path.join(existing_runs[-1], "training_log.json")
                if os.path.exists(last_log):
                    try:
                        with open(last_log, "r", encoding="utf-8") as f:
                            meta = json.load(f).get("model_metadata", {})
                            raw_time = meta.get("end_time") or meta.get("start_time")
                            if raw_time:
                                archive_timestamp = raw_time.replace(" ", "_").replace(":", "-")
                    except Exception:
                        pass
                if not archive_timestamp:
                    archive_timestamp = datetime.fromtimestamp(os.path.getmtime(existing_runs[-1])).strftime("%Y-%m-%d_%H-%M-%S")

                archive_target_dir = os.path.join(self.base_runs_dir, "_archive", archive_timestamp)
                os.makedirs(archive_target_dir, exist_ok=True)
                for r_path in existing_runs:
                    shutil.move(r_path, os.path.join(archive_target_dir, os.path.basename(r_path)))
                print(f"✓ Eski koşuların tamamı arşivlendi ➔ {archive_target_dir}")
                existing_runs = []
            print("!" * 70 + "\n")

        # Eğer dışarıdan açıkça run_tag verildiyse onu kullan, verilmediyse otomatik hesapla
        if run_tag is not None:
            self.run_tag = run_tag
        else:
            run_idx = len(existing_runs) + 1
            self.run_tag = f"run{run_idx}"

        self.runs_dir = os.path.join(self.base_runs_dir, self.run_tag)
        os.makedirs(self.runs_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file_path = os.path.join(self.runs_dir, "training_log.json")

        self.data = {
            "model_metadata": {
                "model_family": self.model_family,
                "run_tag": self.run_tag,
                "start_time": self.start_datetime,
                "end_time": None,
                "completed": False,
                "best_val_loss": None,
                "total_steps": total_steps
            },
            "hyperparameters": config,
            "system_info": {
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "torch_version": torch.__version__
            },
            "history": []
        }
        self._write_to_disk()
        self._print_header()

    def _get_gpu_telemetry(self):
        if not self.has_nvml: return None, None
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
            return temp, util
        except Exception:
            return None, None

    def _print_header(self):
        print("\n" + "=" * 120)
        print(f"🔬 DORINA LAB: {self.model_family} ➔ {self.run_tag.upper()}")
        print(f"⚡ Donanım: {self.data['system_info']['device']} | Hedef: {self.total_steps} Adım")
        print("=" * 120)
        print(f"{'Adım':<9} | {'Train':<7} | {'Val':<7} | {'Perp':<7} | {'Tok/s':<8} | {'VRAM(MB)':<9} | {'GPU(%)':<7} | {'Isı(°C)':<7} | {'Geçen':<9} | {'Kalan'}")
        print("-" * 120)

    def log(self, step: int, train_loss: float, val_loss: float = None, grad_norm: float = None, lr: float = 0.0, batch_tokens: int = 0, sample_text: str = ""):
        elapsed_total = time.time() - self.start_time
        step_duration = time.time() - self.step_start_time
        self.step_start_time = time.time()

        eval_loss = val_loss if val_loss is not None else train_loss
        perplexity = round(math.exp(min(eval_loss, 20)), 2)

        is_best = False
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = round(float(val_loss), 4)
            self.data["model_metadata"]["best_val_loss"] = self.best_val_loss
            is_best = True

        tokens_per_sec = round(batch_tokens / step_duration, 1) if step_duration > 0 and batch_tokens > 0 else 0
        vram_used_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else 0
        gpu_temp, gpu_util = self._get_gpu_telemetry()

        elapsed_str = f"{elapsed_total/60:.1f} dk" if elapsed_total >= 60 else f"{elapsed_total:.1f} sn"
        steps_left = self.total_steps - step
        eta_seconds = steps_left * (elapsed_total / max(step, 1))
        eta_str = f"{eta_seconds/60:.1f} dk" if eta_seconds >= 60 else f"{int(eta_seconds)} sn"
        if steps_left <= 0: eta_str = "Bitti"

        val_str = f"{val_loss:.4f}" if val_loss is not None else "---"
        util_str = f"%{gpu_util}" if gpu_util is not None else "---"
        temp_str = f"{gpu_temp}°C" if gpu_temp is not None else "---"
        best_marker = " ⭐" if is_best else ""

        print(f"[{step:4d}/{self.total_steps}] | {train_loss:<7.4f} | {val_str:<7} | {perplexity:<7.2f} | {tokens_per_sec:<8} | {vram_used_mb:<9} | {util_str:<7} | {temp_str:<7} | {elapsed_str:<9} | {eta_str}{best_marker}")

        if sample_text:
            clean_sample = sample_text.replace('\n', ' ')[:95]
            print(f"   ↳ Örnek: \"{clean_sample}...\"")
            print("-" * 120)

        self.data["history"].append({
            "step": step, "train_loss": round(float(train_loss), 4),
            "val_loss": round(float(val_loss), 4) if val_loss is not None else None,
            "perplexity": perplexity, "grad_norm": grad_norm, "learning_rate": lr,
            "tokens_per_sec": tokens_per_sec, "vram_allocated_mb": vram_used_mb,
            "gpu_temperature_c": gpu_temp, "gpu_utilization_pct": gpu_util,
            "elapsed_seconds": round(elapsed_total, 2)
        })
        self._write_to_disk()

    def finish(self, model, encode_fn, decode_fn, test_prompts: list, device: str):
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_time = round(time.time() - self.start_time, 2)

        self.data["model_metadata"]["end_time"] = end_datetime
        self.data["model_metadata"]["completed"] = True
        self.data["model_metadata"]["total_elapsed_seconds"] = total_time

        # 1. runs/ altına model.pt ve best.pt kaydet
        local_model_path = os.path.join(self.runs_dir, "model.pt")
        torch.save(model.state_dict(), local_model_path)

        # 2. Test Çıktıları
        model.eval()
        samples_file_path = os.path.join(self.runs_dir, "test_samples.txt")
        final_test_results = {}
        with open(samples_file_path, "w", encoding="utf-8") as sf:
            sf.write(f"DORINA AI - TEST SAMPLES\nModel: {self.model_family} ({self.run_tag})\nTarih: {end_datetime}\n" + "="*60 + "\n\n")
            for prompt in test_prompts:
                tokens = encode_fn(prompt)
                ctx = torch.tensor([tokens], dtype=torch.long, device=device)
                out_tokens = model.generate(ctx, max_new_tokens=60)[0].tolist()
                gen_text = decode_fn(out_tokens)
                final_test_results[prompt] = gen_text
                sf.write(f"[PROMPT]: {prompt}\n[ÇIKTI ]: {gen_text}\n" + "-"*40 + "\n")

        self.data["final_generation_tests"] = final_test_results
        self._write_to_disk()

        # 3. summary.txt
        summary_file_path = os.path.join(self.runs_dir, "summary.txt")
        last_h = self.data["history"][-1] if self.data["history"] else {}
        with open(summary_file_path, "w", encoding="utf-8") as sumf:
            sumf.write("=" * 50 + "\n" + f"DORINA AI TRAINING SUMMARY\n" + "=" * 50 + "\n")
            sumf.write(f"Model Ailesi   : {self.model_family}\nKoşu           : {self.run_tag}\n")
            sumf.write(f"Bitiş          : {end_datetime}\nEn İyi Val     : {self.best_val_loss}\n")
            sumf.write(f"Son Perplexity : {last_h.get('perplexity')}\n" + "=" * 50 + "\n")

        # 4. Grafikleri Üret
        print("\n📊 Grafikler üretiliyor...")
        try:
            from src.utils.plotter import generate_model_dashboard
            generate_model_dashboard(self.runs_dir)
        except Exception as e:
            print(f"⚠️ Grafik çiziminde hata: {e}")

        # 5. OTOMATİK RELEASE / ZAMAN KAPSÜLÜ ADIMI
        self._create_release(model, final_test_results)

    def _create_release(self, model, sample_results):
        print("\n🏛️ KALICI ARŞİV OLUŞTURULUYOR: releases/...")
        release_dir = os.path.join("releases", self.model_family)
        os.makedirs(release_dir, exist_ok=True)

        # A) Kendi Kendine Yeten (Self-Contained) Model Dosyası
        release_model_path = os.path.join(release_dir, "model.pt")
        # En iyi ağırlık varsa onu al, yoksa mevcut ağırlığı koy
        best_ckpt = os.path.join(self.runs_dir, "best.pt")
        weights_to_save = torch.load(best_ckpt, map_location="cpu") if os.path.exists(best_ckpt) else model.state_dict()

        package = {
            "model_state_dict": weights_to_save,
            "config": self.config,
            "tokenizer": "gpt2",
            "model_family": self.model_family,
            "best_val_loss": self.best_val_loss,
            "release_date": self.data["model_metadata"]["end_time"]
        }
        torch.save(package, release_model_path)
        print(f"  ✓ Model paketi hazırlandı ➔ {release_model_path}")

        # B) Config JSON
        with open(os.path.join(release_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        # C) Data Manifest
        manifest_src = "data/data_manifest.json"
        if os.path.exists(manifest_src):
            shutil.copy(manifest_src, os.path.join(release_dir, "data_manifest.json"))

        # D) Kod Anlık Görüntüsü (Code Snapshot)
        code_dir = os.path.join(release_dir, "code")
        os.makedirs(code_dir, exist_ok=True)
        for py_file in ["train.py", "prepare_data.py"]:
            if os.path.exists(py_file): shutil.copy(py_file, code_dir)
        if os.path.exists("src/model.py"): shutil.copy("src/model.py", code_dir)
        if os.path.exists("src/utils/logger.py"): shutil.copy("src/utils/logger.py", code_dir)

        # E) Raporlar ve Grafikler Kopyası
        for fn in ["training_log.json", "summary.txt", "test_samples.txt"]:
            src_p = os.path.join(self.runs_dir, fn)
            if os.path.exists(src_p): shutil.copy(src_p, release_dir)
        plots_src = os.path.join(self.runs_dir, "plots")
        plots_dst = os.path.join(release_dir, "plots")
        if os.path.exists(plots_src):
            if os.path.exists(plots_dst): shutil.rmtree(plots_dst)
            shutil.copytree(plots_src, plots_dst)

        # F) Otomatik README.md
        git_hash = "N/A"
        try:
            git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
        except Exception:
            pass

        readme_content = f"""# {self.model_family} - Resmi Sürüm (Release)

- **Tarih:** {self.data['model_metadata']['end_time']}
- **Git Commit:** `{git_hash}`
- **En İyi Val Loss:** {self.best_val_loss}
- **Mimari:** {self.config.get('n_layer')} Katman | {self.config.get('n_embd')} Embedding | {self.config.get('n_head')} Head
- **Kontekst Uzunluğu:** {self.config.get('context_length')}
- **Toplam Eğitim Adımı:** {self.total_steps}

## Örnek Üretimler:
"""
        for p, out in sample_results.items():
            readme_content += f"\n**Prompt:** `{p}`\n> {out}\n"

        with open(os.path.join(release_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)

        print(f"  ✓ README.md ve Kod kopyaları tamamlandı ➔ {release_dir}")

    def _write_to_disk(self):
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)