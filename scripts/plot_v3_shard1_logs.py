import re
import matplotlib.pyplot as plt

def parse_and_plot(log_path, output_png):
    steps = []
    losses = []
    lrs = []
    speeds = []
    
    val_steps = []
    val_losses = []
    
    moe_steps = []
    dropped_pcts = []
    dead_experts = []
    entropies = []

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Step parse
    step_pattern = re.compile(r"Adım\s+(\d+)/1500\s+\|\s+Loss:\s+([\d\.]+)\s+\|\s+LR:\s+([\d\.e\-]+)\s+\|\s+Hız:\s+([\d,]+)\s+tok/s")
    for match in step_pattern.finditer(content):
        s = int(match.group(1))
        l = float(match.group(2))
        lr = float(match.group(3))
        spd = float(match.group(4).replace(",", ""))
        steps.append(s)
        losses.append(l)
        lrs.append(lr)
        speeds.append(spd)

    # Val parse
    val_pattern = re.compile(r">>>\s+\[EVAL Adım\s+(\d+)\]\s+Val Loss:\s+([\d\.]+)")
    for match in val_pattern.finditer(content):
        val_steps.append(int(match.group(1)))
        val_losses.append(float(match.group(2)))

    # MoE parse
    moe_pattern = re.compile(r"\[MoE Sağlık\]\s+Düşme:\s+%([\d\.]+)\s+\|\s+Ölü Uzman:\s+(\d+)\s+\|\s+Router Entropi:\s+([\d\.]+)")
    matches = list(moe_pattern.finditer(content))
    for idx, match in enumerate(matches):
        moe_steps.append((idx + 1) * 50)
        dropped_pcts.append(float(match.group(1)))
        dead_experts.append(int(match.group(2)))
        entropies.append(float(match.group(3)))

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Dorina V3 (180.6M MoE) - Shard 1 (200M Token) Eğitim Raporu', fontsize=16, fontweight='bold')

    # 1. Loss Eğrisi
    axs[0, 0].plot(steps, losses, label='Train Loss', color='#1f77b4', alpha=0.8, linewidth=1.5)
    if val_steps:
        axs[0, 0].plot(val_steps, val_losses, 'ro-', label='Val Loss (EVAL)', linewidth=2)
    axs[0, 0].set_title('Kayıp (Loss) Eğrisi: 10.51 -> 1.51', fontsize=13)
    axs[0, 0].set_xlabel('Adım')
    axs[0, 0].set_ylabel('Cross-Entropy Loss')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].legend()

    # 2. LR Programı (WSD)
    axs[0, 1].plot(steps, lrs, label='Learning Rate', color='#2ca02c', linewidth=2)
    axs[0, 1].set_title('WSD Öğrenme Oranı Programı', fontsize=13)
    axs[0, 1].set_xlabel('Adım')
    axs[0, 1].set_ylabel('LR')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].legend()

    # 3. MoE Uzman Sağlığı
    ax3 = axs[1, 0]
    ax3.plot(moe_steps, dead_experts, 'r^-', label='Ölü Uzman Sayısı (32 Üzerinden)', linewidth=2)
    ax3.set_title('MoE Uzman Katılımı & Ölü Uzmanlar', fontsize=13)
    ax3.set_xlabel('Adım')
    ax3.set_ylabel('Ölü Uzman', color='r')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper right')

    # 4. Token Hızı
    axs[1, 1].plot(steps, speeds, label='Gerçekleşen Hız (tok/s)', color='#ff7f0e', linewidth=1.5)
    axs[1, 1].set_title(f'Donanım Verimi: Ortalama ~{sum(speeds)/len(speeds):,.0f} tok/s', fontsize=13)
    axs[1, 1].set_xlabel('Adım')
    axs[1, 1].set_ylabel('Tokens / Saniye')
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"Grafik kaydedildi: {output_png}")

if __name__ == "__main__":
    parse_and_plot("data/v3_shard1_kaggle_raw_log.txt", "data/v3_shard1_training_report.png")
