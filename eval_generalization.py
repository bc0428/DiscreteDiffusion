"""
eval_generalization.py
======================
Evaluates generalization performance on the offline generalization.pt dataset.
Leverages the exact metrics and denoising logic from test_model.py.
Saves statistics into the findings/ directory.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
from pathlib import Path
import csv
import math
from datetime import datetime
import time
import torch
from collections import defaultdict

# Import the core network and corruption wrapper
sys.path.insert(0, str(Path(__file__).parent))
from main import TheoryDenoiserNet, D3PMForwardCorruption

# Import the exact evaluation logic from your test_model.py
from test_model import (
    denoise_until_consistent,
    _extract_theory_from_offline_entry,
    _sample_trajectory_start,
    START_DENOISE_TRAJ_PCT_MIN,
    START_DENOISE_TRAJ_PCT_MAX
)

# Configuration
CHECKPOINT_PATH = "outputs/theory_denoiser_rl_finetuned.pt"  # Make sure this points to your checkpoint
OUTPUT_DIR = Path("findings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

log_lines: list[str] = []


def emit(msg: str = "") -> None:
    """Print to console and capture lines for a log file."""
    print(msg)
    log_lines.append(msg)

torch.manual_seed(SEED)

def main():

    log_path = OUTPUT_DIR / f"eval_generalization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emit(f"Loading checkpoint from: {CHECKPOINT_PATH} to {device}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    T = cfg["NUM_TIMESTEPS"]
    K = cfg["K_STATES"]

    # In generalization, we expect max sizes up to double the original training limits
    max_N = cfg["N_LITERALS"] * 2
    max_M = cfg["M_CLAUSES"] * 2

    model = TheoryDenoiserNet(N=max_N, M=max_M, num_classes=K, num_timesteps=T)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    corrupt = D3PMForwardCorruption(num_classes=K, num_timesteps=T).to(device)

    emit(f"\nProcessing generalization dataset files one by one...")

    # Grab all files matching the new naming pattern
    chunk_files = sorted(Path("dataset").glob("generalization_N*_M*.pt"))

    if not chunk_files:
        emit("ERROR: No generalization files found in dataset/ folder. Run generator first.")
        sys.exit(1)

    results = []
    all_theory_sizes = {}  # Track all theory sizes and their metrics

    # Process each chunk file individually to prevent RAM overflow
    for chunk_file in chunk_files:
        emit(f"\n{'='*60}")
        emit(f"Reading and evaluating: {chunk_file.name}...")
        emit(f"{'='*60}")

        chunk_data = torch.load(chunk_file, map_location="cpu", weights_only=False)

        # Group theories from this chunk by their N and M size
        grouped_theories = defaultdict(list)
        for entry in chunk_data:
            theory = _extract_theory_from_offline_entry(entry)
            start_t = _sample_trajectory_start(entry, T, START_DENOISE_TRAJ_PCT_MIN, START_DENOISE_TRAJ_PCT_MAX)
            n_size = int(entry.get("original_n", theory.size(0)))
            m_size = int(entry.get("original_m", theory.size(1)))

            # Ensure we don't accidentally exceed tensor bounds if malformed
            actual_n = max(1, min(n_size, theory.size(0)))
            grouped_theories[(actual_n, m_size)].append((theory, start_t))

        # Evaluate this chunk's groups
        for (N, M), theories in grouped_theories.items():
            theory_size = N * M
            emit(f"\n  Theory Size N={N}, M={M} (N*M={theory_size}) | Samples in file: {len(theories)}")

            # Initialize or retrieve metrics for this theory size
            if (N, M) not in all_theory_sizes:
                all_theory_sizes[(N, M)] = {
                    "N": N, "M": M, "theory_size": theory_size,
                    "input_consistent": [],
                    "consistent": [],
                    "solve_time_sec": [],
                    "norm_steps": [],
                    "total_change": [],
                    "empty_change": []
                }

            metrics = all_theory_sizes[(N, M)]

            for idx, (theory, start_t) in enumerate(theories):
                t0 = time.perf_counter()
                res = denoise_until_consistent(
                    theory=theory,
                    model=model,
                    corrupt=corrupt,
                    start_t=start_t,
                    device=device,
                )
                solve_time_sec = time.perf_counter() - t0

                metrics["input_consistent"].append(1 if res["input_consistent"] else 0)
                metrics["consistent"].append(1 if res["consistent"] else 0)
                metrics["solve_time_sec"].append(solve_time_sec)
                metrics["total_change"].append(res["total_change_pct"])
                metrics["empty_change"].append(res["empty_change_pct"])

                if res["consistent"]:
                    metrics["norm_steps"].append((res["steps_to_first"] / start_t) * 100.0)

                if (idx + 1) % max(1, len(theories) // 5) == 0:
                    emit(f"    Processed {idx + 1}/{len(theories)}")

        # Clear chunk data to free memory
        del chunk_data
        del grouped_theories
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compute final statistics for all theory sizes
    emit(f"\n{'='*60}")
    emit("Computing final statistics across all files...")
    emit(f"{'='*60}")

    def get_stats(arr):
        if not arr: return 0.0, 0.0
        mean_val = sum(arr) / len(arr)
        var_val = sum((x - mean_val)**2 for x in arr) / (len(arr) - 1) if len(arr) > 1 else 0.0
        return mean_val, math.sqrt(var_val)

    for (N, M), metrics in sorted(all_theory_sizes.items()):
        ic_mean, ic_std = get_stats(metrics["input_consistent"])
        c_mean, c_std = get_stats(metrics["consistent"])
        st_mean, st_std = get_stats(metrics["solve_time_sec"])
        ns_mean, ns_std = get_stats(metrics["norm_steps"])
        tc_mean, tc_std = get_stats(metrics["total_change"])
        ec_mean, ec_std = get_stats(metrics["empty_change"])

        ic_mean_pct = ic_mean * 100.0
        ic_std_pct = ic_std * 100.0
        c_mean_pct = c_mean * 100.0
        c_std_pct = c_std * 100.0

        total_samples = len(metrics["consistent"])
        emit(f"\n--- Theory Size N={N}, M={M} (N*M={metrics['theory_size']}) | Total Samples: {total_samples} ---")
        emit(f"  Input Consistent Rate: {ic_mean_pct:.1f}% ± {ic_std_pct:.1f}%")
        emit(f"  Consistency Rate: {c_mean_pct:.1f}% ± {c_std_pct:.1f}%")
        emit(f"  Solve Time/Theory (s): {st_mean:.4f} ± {st_std:.4f}")
        emit(f"  Norm Steps (Avg %): {ns_mean:.1f}% ± {ns_std:.1f}%")
        emit(f"  Total Change (%): {tc_mean:.1f}% ± {tc_std:.1f}%")

        results.append({
            "N": N, "M": M, "theory_size": metrics["theory_size"],
            "total_samples": total_samples,
            "input_consistent_rate_mean": ic_mean_pct, "input_consistent_rate_std": ic_std_pct,
            "consistency_rate_mean": c_mean_pct, "consistency_rate_std": c_std_pct,
            "solve_time_mean_sec": st_mean, "solve_time_std_sec": st_std,
            "norm_steps_mean": ns_mean, "norm_steps_std": ns_std,
            "total_change_mean": tc_mean, "total_change_std": tc_std,
            "empty_change_mean": ec_mean, "empty_change_std": ec_std
        })

    csv_path = OUTPUT_DIR / "generalization_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    emit(f"\nStats successfully saved to {csv_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    emit(f"Log saved to {log_path}")

if __name__ == "__main__":
    main()