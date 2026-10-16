import os
import glob
import json
import matplotlib
matplotlib.use('Agg')  # Headless arka plan çizimi
import matplotlib.pyplot as plt
import numpy as np

def compare_with_previous_run(current_run_dir):
    """Aynı model klasörü altındaki bir önceki run ile kıyaslama yapar."""
    run_folder_name = os.path.basename(os.path.normpath(current_run_dir))
    family_dir = os.path.dirname(current_run_dir)
    family_name = os.path.basename(family_dir)

    if not run_folder_name.startswith("run"):
        return

    try:
        current_idx = int(run_folder_name.replace("run", ""))
    except ValueError:
        return

    # run1 ise kıyaslayacak önceki koşu yoktur
    if current_idx <= 1:
        return

    prev_idx = current_idx - 1
    prev_run_dir = os.path.join(family_dir, f"run{prev_idx}")
    
    # Standart tek isim: training_log.json
    prev_json = os.path.join(prev_run_dir, "training_log.json")
    curr_json = os.path.join(current_run_dir, "training_log.json")

    if not (os.path.exists(prev_json) and os.path.exists(curr_json)):
        return

    with open(prev_json, "r", encoding="utf-8") as f:
        prev_data = json.load(f)
    with open(curr_json, "r", encoding="utf-8") as f:
        curr_data = json.load(f)

    p_hist = prev_data.get("history", [])
    c_hist = curr_data.get("history", [])
    if not p_hist or not c_hist:
        return

    plots_dir = os.path.join(current_run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    p_steps = [e["step"] for e in p_hist]
    p_train = [e.get("train_loss") for e in p_hist]
    p_val_steps = [e["step"] for e in p_hist if e.get("val_loss") is not None]
    p_val = [e.get("val_loss") for e in p_hist if e.get("val_loss") is not None]
    p_perp_steps = [e["step"] for e in p_hist if e.get("perplexity") is not None]
    p_perp = [e.get("perplexity") for e in p_hist if e.get("perplexity") is not None]
    p_temp = [e.get("gpu_temperature_c") for e in p_hist if e.get("gpu_temperature_c") is not None]
    p_speed = [e.get("tokens_per_sec") for e in p_hist if e.get("tokens_per_sec") is not None]

    c_steps = [e["step"] for e in c_hist]
    c_train = [e.get("train_loss") for e in c_hist]
    c_val_steps = [e["step"] for e in c_hist if e.get("val_loss") is not None]
    c_val = [e.get("val_loss") for e in c_hist if e.get("val_loss") is not None]
    c_perp_steps = [e["step"] for e in c_hist if e.get("perplexity") is not None]
    c_perp = [e.get("perplexity") for e in c_hist if e.get("perplexity") is not None]
    c_temp = [e.get("gpu_temperature_c") for e in c_hist if e.get("gpu_temperature_c") is not None]
    c_speed = [e.get("tokens_per_sec") for e in c_hist if e.get("tokens_per_sec") is not None]

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    fig.suptitle(f"DORINA LAB - GELİŞİM ANALİZİ: run{prev_idx} vs run{current_idx}\n({family_name})", fontsize=15, fontweight="bold")

    max_p_step = p_steps[-1] if p_steps else 0
    c_cont_steps = [s + max_p_step for s in c_steps]
    c_val_cont_steps = [s + max_p_step for s in c_val_steps]

    axes[0, 0].plot(p_steps, p_train, label=f"run{prev_idx} (Önceki)", color="#1f77b4", lw=1.5)
    axes[0, 0].plot(c_cont_steps, c_train, label=f"run{current_idx} (Mevcut)", color="#2ca02c", lw=2)
    if p_val and c_val:
        axes[0, 0].plot(p_val_steps, p_val, 'o--', color="#ff7f0e", markersize=3, label=f"run{prev_idx} Val")
        axes[0, 0].plot(c_val_cont_steps, c_val, 's--', color="#d62728", markersize=4, label=f"run{current_idx} Val")
    axes[0, 0].axvline(x=max_p_step, color="red", linestyle=":", lw=1.5, label="Devam Noktası")
    axes[0, 0].set_title("1. Kesintisiz Model Yaşam Döngüsü")
    axes[0, 0].set_xlabel("Kümülatif Adım")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(fontsize=8)

    min_loss = min(min(p_train), min(c_train))
    focus_max = max(p_train[-1], c_train[0]) * 1.25
    axes[0, 1].plot(p_steps, p_train, ":", label=f"run{prev_idx} Train", color="#1f77b4", alpha=0.5)
    axes[0, 1].plot(c_steps, c_train, "-", label=f"run{current_idx} Train (Yeni)", color="#1f77b4", lw=2)
    if p_val:
        axes[0, 1].plot(p_val_steps, p_val, ":o", label=f"run{prev_idx} Val", color="#ff7f0e", alpha=0.5, markersize=3)
    if c_val:
        axes[0, 1].plot(c_val_steps, c_val, "-o", label=f"run{current_idx} Val (Yeni)", color="#d62728", lw=2, markersize=4)
    axes[0, 1].set_ylim([max(0, min_loss * 0.95), focus_max])
    axes[0, 1].set_title(f"2. Yakınlaştırılmış Loss ({min_loss:.2f} - {focus_max:.2f})")
    axes[0, 1].set_xlabel("Adım (Step)")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(fontsize=8)

    if p_perp and c_perp:
        axes[1, 0].plot(p_perp_steps, p_perp, ":", label=f"run{prev_idx}", color="#7f7f7f", lw=1.5)
        axes[1, 0].plot(c_perp_steps, c_perp, "-s", label=f"run{current_idx} (Yeni)", color="#2ca02c", lw=2, markersize=4)
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_title("3. Perplexity (Logaritmik)")
        axes[1, 0].set_xlabel("Adım")
        axes[1, 0].set_ylabel("Perplexity")
        axes[1, 0].grid(True, which="both", linestyle="--", alpha=0.5)
        axes[1, 0].legend(fontsize=8)

    min_len = min(len(p_train), len(c_train))
    delta_loss = [p_train[i] - c_train[i] for i in range(min_len)]
    axes[1, 1].plot(c_steps[:min_len], delta_loss, color="#9467bd", lw=2)
    axes[1, 1].axhline(y=0, color="black", linestyle="--", alpha=0.7)
    axes[1, 1].fill_between(c_steps[:min_len], 0, delta_loss, where=(np.array(delta_loss) >= 0), color="#2ca02c", alpha=0.3, label="Gelişme")
    axes[1, 1].fill_between(c_steps[:min_len], 0, delta_loss, where=(np.array(delta_loss) < 0), color="#d62728", alpha=0.3, label="Gerileme")
    axes[1, 1].set_title("4. Net Fark: run(eski) - run(yeni)")
    axes[1, 1].set_xlabel("Adım")
    axes[1, 1].set_ylabel("İyileşme (Delta)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(fontsize=8)

    if p_temp and c_temp:
        axes[2, 0].plot(p_steps[:len(p_temp)], p_temp, ":", label=f"run{prev_idx} Isı", color="#ff7f0e", lw=1.5)
        axes[2, 0].plot(c_steps[:len(c_temp)], c_temp, "-", label=f"run{current_idx} Isı", color="#d62728", lw=2)
        axes[2, 0].set_title("5. GPU Sıcaklık (°C)")
        axes[2, 0].set_xlabel("Adım")
        axes[2, 0].set_ylabel("°C")
        axes[2, 0].grid(True, linestyle="--", alpha=0.5)
        axes[2, 0].legend(fontsize=8)

    if p_speed and c_speed:
        axes[2, 1].plot(p_steps[:len(p_speed)], p_speed, ":", label=f"run{prev_idx} Hız", color="#1f77b4", lw=1.5)
        axes[2, 1].plot(c_steps[:len(c_speed)], c_speed, "-", label=f"run{current_idx} Hız", color="#17becf", lw=2)
        axes[2, 1].set_title("6. Hız (Token / sn)")
        axes[2, 1].set_xlabel("Adım")
        axes[2, 1].set_ylabel("Token/sn")
        axes[2, 1].grid(True, linestyle="--", alpha=0.5)
        axes[2, 1].legend(fontsize=8)

    plt.tight_layout()
    cmp_path = os.path.join(plots_dir, "run_comparison.png")
    plt.savefig(cmp_path, dpi=250)
    plt.close()
    print(f"  📊 Kıyaslama grafiği üretildi -> {cmp_path}")


def generate_model_dashboard(run_dir):
    """Herhangi bir run klasörünün içindeki grafikleri üretir."""
    json_path = os.path.join(run_dir, "training_log.json")
    if not os.path.exists(json_path):
        print(f"⚠️ 'training_log.json' bulunamadı: {run_dir}")
        return

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("model_metadata", {})
    model_family = meta.get("model_family") or meta.get("family") or "Model"
    run_tag = meta.get("run_tag") or meta.get("run") or os.path.basename(run_dir)
    model_title = f"{model_family} ({run_tag})"

    history = data.get("history", [])
    if not history:
        return

    steps = [e["step"] for e in history]
    train_losses = [e.get("train_loss") for e in history]
    val_losses = [e.get("val_loss") for e in history if e.get("val_loss") is not None]
    val_steps = [e["step"] for e in history if e.get("val_loss") is not None]
    perplexities = [e.get("perplexity") for e in history if e.get("perplexity") is not None]
    perp_steps = [e["step"] for e in history if e.get("perplexity") is not None]
    gpu_temps = [e.get("gpu_temperature_c") for e in history if e.get("gpu_temperature_c") is not None]
    temp_steps = [e["step"] for e in history if e.get("gpu_temperature_c") is not None]
    gpu_utils = [e.get("gpu_utilization_pct") for e in history if e.get("gpu_utilization_pct") is not None]
    util_steps = [e["step"] for e in history if e.get("gpu_utilization_pct") is not None]
    vrams = [e.get("vram_allocated_mb") for e in history if e.get("vram_allocated_mb") is not None]
    vram_steps = [e["step"] for e in history if e.get("vram_allocated_mb") is not None]

    # 1. Loss
    plt.figure(figsize=(9, 5))
    plt.plot(steps, train_losses, label="Train Loss", color="#1f77b4", lw=2)
    if val_losses:
        plt.plot(val_steps, val_losses, label="Val Loss", color="#ff7f0e", marker="o", lw=1.5)
    plt.title(f"{model_title} - Kayıp (Loss) Eğrisi", fontweight="bold")
    plt.xlabel("Adım (Step)")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=250)
    plt.close()

    # 2. Perplexity
    if perplexities:
        plt.figure(figsize=(9, 5))
        plt.plot(perp_steps, perplexities, color="#2ca02c", lw=2, marker="s", markersize=3)
        plt.title(f"{model_title} - Perplexity", fontweight="bold")
        plt.xlabel("Adım (Step)")
        plt.ylabel("Perplexity")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "perplexity_curve.png"), dpi=250)
        plt.close()

    # 3. Telemetry
    if gpu_temps or vrams:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        if gpu_temps:
            ax1.plot(temp_steps, gpu_temps, color="#d62728", label="Sıcaklık (°C)", lw=2)
            ax1.set_ylabel("Sıcaklık (°C)", color="#d62728")
        ax1.set_xlabel("Adım (Step)")
        ax1.grid(True, linestyle="--", alpha=0.5)

        if vrams:
            ax2 = ax1.twinx()
            ax2.plot(vram_steps, vrams, color="#9467bd", linestyle="--", label="VRAM (MB)", lw=2)
            ax2.set_ylabel("VRAM (MB)", color="#9467bd")

        plt.title(f"{model_title} - Donanım Telemetrisi", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "hardware_telemetry.png"), dpi=250)
        plt.close()

    # 4. 4'lü Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"DORINA LAB RAPORU: {model_title}", fontsize=14, fontweight="bold")
    axes[0, 0].plot(steps, train_losses, label="Train Loss", color="#1f77b4")
    if val_losses:
        axes[0, 0].plot(val_steps, val_losses, label="Val Loss", color="#ff7f0e", marker="o", markersize=3)
    axes[0, 0].set_title("Kayıp (Loss)")
    axes[0, 0].set_xlabel("Adım")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()

    if perplexities:
        axes[0, 1].plot(perp_steps, perplexities, color="#2ca02c")
        axes[0, 1].set_title("Perplexity")
        axes[0, 1].set_xlabel("Adım")
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    if gpu_temps:
        axes[1, 0].plot(temp_steps, gpu_temps, color="#d62728", label="Isı (°C)")
    if gpu_utils:
        axes[1, 0].plot(util_steps, gpu_utils, color="#ff7f0e", linestyle=":", label="GPU %")
    axes[1, 0].set_title("GPU Yükü ve Isısı")
    axes[1, 0].set_xlabel("Adım")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend()

    if vrams:
        axes[1, 1].plot(vram_steps, vrams, color="#9467bd", label="VRAM (MB)")
    axes[1, 1].set_title("Bellek (VRAM)")
    axes[1, 1].set_xlabel("Adım")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "dashboard_summary.png"), dpi=250)
    plt.close()
    print(f"  ✓ Grafikler başarıyla üretildi -> {plots_dir}")

    # Önceki koşu varsa onunla kıyasla
    compare_with_previous_run(run_dir)


if __name__ == "__main__":
    for family_folder in glob.glob("runs/*"):
        if os.path.isdir(family_folder) and not os.path.basename(family_folder).startswith("_"):
            for run_folder in glob.glob(os.path.join(family_folder, "run*")):
                if os.path.isdir(run_folder):
                    generate_model_dashboard(run_folder)