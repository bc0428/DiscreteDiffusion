"""
plot_generalization.py
======================
Plots the 3 main metrics from the generalization evaluations:
1. Consistency Rate
2. Normalized Denoising Steps %
3. Revision Percentages (Total & Empty Changes)
Saves each graph into the findings/ directory.
"""

import sys
from pathlib import Path
import csv

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)

def load_csv(csv_path):
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({k: float(v) for k, v in row.items()})
    return sorted(results, key=lambda x: x["theory_size"])

def create_plot(x, y, yerr, title, ylabel, output_path, y2=None, yerr2=None, label1=None, label2=None):
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.errorbar(x, y, yerr=yerr, marker='o', markersize=6, linestyle='-', linewidth=2, label=label1, capsize=4)
    if y2 is not None:
        ax.errorbar(x, y2, yerr=yerr2, marker='s', markersize=6, linestyle='--', linewidth=2, label=label2, capsize=4)
        ax.legend(fontsize=11)

    ax.set_xlabel("Theory Size (N × M)", fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()

def main():
    findings_dir = Path("findings")
    csv_path = findings_dir / "generalization_stats.csv"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Please run eval_generalization.py first.")
        sys.exit(1)

    data = load_csv(csv_path)
    sizes = [d["theory_size"] for d in data]

    # Plot 1: Consistency Rate
    create_plot(
        x=sizes,
        y=[d["consistency_rate_mean"] for d in data],
        yerr=[d["consistency_rate_std"] for d in data],
        title="Generalization: Consistency Rate vs Theory Size",
        ylabel="Consistency Rate (%)",
        output_path=findings_dir / "plot_consistency_rate.png"
    )

    # Plot 2: Normalized Denoising Steps %
    create_plot(
        x=sizes,
        y=[d["norm_steps_mean"] for d in data],
        yerr=[d["norm_steps_std"] for d in data],
        title="Generalization: Normalized Denoising Steps vs Theory Size",
        ylabel="Steps to Consistency (% of Starting Steps)",
        output_path=findings_dir / "plot_denoising_steps.png"
    )

    # Plot 3: Revision Metrics (Total vs Empty Change)
    create_plot(
        x=sizes,
        y=[d["total_change_mean"] for d in data],
        yerr=[d["total_change_std"] for d in data],
        y2=[d["empty_change_mean"] for d in data],
        yerr2=[d["empty_change_std"] for d in data],
        label1="Total Change (%)",
        label2="Empty Change (%)",
        title="Generalization: Revision Impact vs Theory Size",
        ylabel="Percentage Change (%)",
        output_path=findings_dir / "plot_revision_pct.png"
    )

if __name__ == "__main__":
    main()