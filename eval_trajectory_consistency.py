import argparse
from pathlib import Path

import torch

from main import is_consistent

try:
    import generate_dataset as gd
except ModuleNotFoundError:
    # Workspace currently has generate_dataset_parallel.py in place of generate_dataset.py.
    import generate_dataset_parallel as gd

PCT_BUCKETS = list(range(0, 101, 10))


def _bucket_indices(traj_len: int) -> list[int]:
    max_idx = max(0, traj_len - 1)
    out: list[int] = []
    for pct in PCT_BUCKETS:
        idx = int(round((pct / 100.0) * max_idx))
        out.append(min(max(idx, 0), max_idx))
    return out


def _eval_split(samples: list[dict], split_name: str, max_samples: int | None = None) -> dict[int, float]:
    if max_samples is not None:
        samples = samples[:max_samples]

    consistent_counts = {pct: 0 for pct in PCT_BUCKETS}
    totals = {pct: 0 for pct in PCT_BUCKETS}

    for sample in samples:
        traj = sample["trajectory"]
        if not isinstance(traj, torch.Tensor) or traj.ndim != 3:
            continue

        original_m = int(sample.get("original_m", traj.shape[2]))
        bucket_idxs = _bucket_indices(traj.shape[0])

        for pct, idx in zip(PCT_BUCKETS, bucket_idxs):
            theory = traj[idx, :, :original_m].cpu()
            totals[pct] += 1
            if theory.numel() > 0 and is_consistent(theory):
                consistent_counts[pct] += 1

    consistent_rates = {}
    unsat_rates = {}
    for pct in PCT_BUCKETS:
        denom = totals[pct]
        consistent_rate = (100.0 * consistent_counts[pct] / denom) if denom > 0 else 0.0
        consistent_rates[pct] = consistent_rate
        unsat_rates[pct] = 100.0 - consistent_rate

    print(f"\n=== {split_name} ===")
    print(f"Samples evaluated: {len(samples)}")
    print("Pct of trajectory | UNSAT theories (%) | Consistent theories (%)")
    print("-" * 42)
    for pct in PCT_BUCKETS:
        print(f"{pct:>3}%             | {unsat_rates[pct]:6.2f}             | {consistent_rates[pct]:6.2f}")

    return unsat_rates


def _load_pt(path: Path) -> list[dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def _make_smoke_samples(n: int) -> list[dict]:
    out: list[dict] = []
    for _ in range(n):
        clean = gd.generate_consistent_theory(
            N=gd.N_LITERALS,
            max_clauses=max(3, gd.M_CLAUSES // 2),
            active_N=gd.N_LITERALS,
            min_active_N=max(1, gd.N_LITERALS // 2),
            min_clauses=3,
        )
        traj = gd.corrupt_theory(clean)

        padded_traj = []
        for st in traj:
            padded = torch.zeros((gd.N_LITERALS, gd.M_CLAUSES), dtype=torch.long)
            padded[:, : st.size(1)] = st
            padded_traj.append(padded)

        out.append(
            {
                "clean": padded_traj[0],
                "corrupted": padded_traj[-1],
                "trajectory": torch.stack(padded_traj),
                "steps": len(padded_traj) - 1,
                "original_m": clean.size(1),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate consistency rate along normalized trajectory progress (0%..100% every 10%)."
    )
    parser.add_argument("--data-dir", type=str, default="dataset", help="Directory containing dataset .pt files")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap per split for faster checks")
    parser.add_argument(
        "--smoke-generate",
        type=int,
        default=0,
        help="Generate N synthetic samples in-memory and evaluate them (no .pt files needed)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["default", "hyperparam", "finetune"],
        default="default",
        help="Type of dataset to evaluate: default (train/val/test), hyperparam (hyperparam_train/val), or finetune (finetune_train/val)",
    )
    args = parser.parse_args()

    if args.smoke_generate > 0:
        smoke = _make_smoke_samples(args.smoke_generate)
        _eval_split(smoke, f"SMOKE ({args.smoke_generate})", max_samples=args.max_samples)
        return

    data_dir = Path(args.data_dir)
    
    # Select split paths based on dataset type
    if args.dataset_type == "default":
        split_paths = {
            "TRAIN": data_dir / "train.pt",
            "VAL": data_dir / "val.pt",
            "TEST": data_dir / "test.pt",
        }
    elif args.dataset_type == "hyperparam":
        split_paths = {
            "HYPERPARAM_TRAIN": data_dir / "hyperparam_train.pt",
            "HYPERPARAM_VAL": data_dir / "hyperparam_val.pt",
        }
    else:  # finetune
        split_paths = {
            "FINETUNE_TRAIN": data_dir / "finetune_train.pt",
            "FINETUNE_VAL": data_dir / "finetune_val.pt",
        }

    missing = [name for name, path in split_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split files in {data_dir}: {missing}. Run generate_dataset_parallel.py first or use --smoke-generate."
        )

    for split_name, path in split_paths.items():
        samples = _load_pt(path)
        _eval_split(samples, split_name, max_samples=args.max_samples)


if __name__ == "__main__":
    main()

