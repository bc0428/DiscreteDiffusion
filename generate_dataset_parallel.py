import torch
import hashlib
import random
import os
import argparse
import concurrent.futures
from pathlib import Path
from pysat.examples.musx import MUSX
from pysat.formula import WCNF
from pysat.solvers import Solver
from tqdm import tqdm
from concurrent.futures.process import BrokenProcessPool

# ==========================================
# Configuration & Hyperparameters
# ==========================================
N_LITERALS = 10
M_CLAUSES = 40
SAT_SOLVER_BACKEND = "m22"

NUM_TRAIN = 8000
NUM_VAL = 1000
NUM_TEST = 1000

EPOCHS = 150
CURRICULUM_N_STAGES = [
    (1, max(1, N_LITERALS // 3)),
    (35, max(1, (2 * N_LITERALS) // 3)),
    (75, N_LITERALS),
]
CURRICULUM_M_STAGES = [
    (1, max(1, M_CLAUSES // 3)),
    (35, max(1, (2 * M_CLAUSES) // 3)),
    (75, M_CLAUSES),
]

OUTPUT_DIR = Path("dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# SAT Helpers
# ==========================================
def theory_to_cnf_with_mapping(theory: torch.Tensor) -> tuple[list[list[int]], list[int]]:
    num_literals, num_clauses = theory.shape
    clauses: list[list[int]] = []
    mapping: list[int] = []

    for j in range(num_clauses):
        clause: list[int] = []
        seen_pos: set[int] = set()
        seen_neg: set[int] = set()

        for i in range(num_literals):
            value = int(theory[i, j].item())
            var_id = i + 1
            if value == 1:
                seen_pos.add(var_id)
                clause.append(var_id)
            elif value == 2:
                seen_neg.add(var_id)
                clause.append(-var_id)

        if seen_pos & seen_neg:
            continue

        if clause:
            clauses.append(clause)
            mapping.append(j)

    return clauses, mapping


def _is_unsat(clauses: list[list[int]]) -> bool:
    with Solver(name=SAT_SOLVER_BACKEND) as solver:
        solver.append_formula(clauses)
        return not solver.solve()


def get_muc_columns(theory: torch.Tensor) -> list[int]:
    clauses, mapping = theory_to_cnf_with_mapping(theory)
    if not clauses:
        return []

    with Solver(name=SAT_SOLVER_BACKEND) as solver:
        solver.append_formula(clauses)
        if solver.solve():
            return []

    wcnf = WCNF()
    for clause in clauses:
        wcnf.append(clause)

    try:
        mus = MUSX(wcnf, solver=SAT_SOLVER_BACKEND).compute()
        if mus is not None:
            return [mapping[idx - 1] for idx in mus]
    except Exception:
        pass

    core = list(range(len(clauses)))
    i = 0
    while i < len(core):
        candidate = core[:i] + core[i + 1:]
        candidate_clauses = [clauses[idx] for idx in candidate]
        if _is_unsat(candidate_clauses):
            core = candidate
        else:
            i += 1

    return [mapping[idx] for idx in core]


def get_satisfying_model(theory: torch.Tensor, cols: list[int]) -> dict[int, bool] | None:
    if not cols:
        return None

    subset = theory[:, cols]
    clauses, _ = theory_to_cnf_with_mapping(subset)
    if not clauses:
        return {}

    with Solver(name=SAT_SOLVER_BACKEND) as solver:
        solver.append_formula(clauses)
        if not solver.solve():
            return None
        model = solver.get_model()
        return {abs(v): (v > 0) for v in model}


# ==========================================
# Theory Generation + Corruption
# ==========================================
def generate_consistent_theory(
        N: int, max_clauses: int, active_N: int, min_active_N: int, min_clauses: int,
) -> torch.Tensor:
    if max_clauses <= 0:
        raise ValueError("max_clauses must be >= 1")

    min_clauses = max(1, min(min_clauses, max_clauses))
    active_N = max(1, min(active_N, N))
    min_active_N = max(1, min(min_active_N, active_N))

    num_clauses = torch.randint(min_clauses, max_clauses + 1, (1,)).item()
    theory = torch.zeros((N, num_clauses), dtype=torch.long)

    num_vars = torch.randint(min_active_N, active_N + 1, (1,)).item()
    active_vars = torch.randperm(active_N)[:num_vars]
    master_assignment = torch.randint(0, 2, (N,), dtype=torch.bool)

    for j in range(num_clauses):
        L = torch.randint(1, num_vars + 1, (1,)).item()
        clause_vars = active_vars[torch.randperm(num_vars)[:L]]
        k = torch.randint(1, L + 1, (1,)).item()

        true_vars = clause_vars[:k]
        false_vars = clause_vars[k:]

        for v in true_vars:
            theory[v, j] = 1 if master_assignment[v] else 2
        for v in false_vars:
            theory[v, j] = 2 if master_assignment[v] else 1

    return theory


def _mutate_clause_against_model(theory: torch.Tensor, clause_j: int, model: dict[int, bool]) -> None:
    N = theory.size(0)
    active_rows = (theory[:, clause_j] != 0).nonzero(as_tuple=False).flatten().tolist()
    if active_rows:
        target_i = active_rows[torch.randint(0, len(active_rows), (1,)).item()]
    else:
        target_i = torch.randint(0, N, (1,)).item()

    var_id = target_i + 1
    model_val = model.get(var_id, True)
    contradictory_lit = 2 if model_val else 1

    theory[:, clause_j] = 0
    theory[target_i, clause_j] = contradictory_lit


def _random_mutation(theory: torch.Tensor, mutable_cols: list[int]) -> bool:
    if not mutable_cols:
        return False
    N = theory.size(0)
    target_j = mutable_cols[torch.randint(0, len(mutable_cols), (1,)).item()]
    target_i = torch.randint(0, N, (1,)).item()
    current = int(theory[target_i, target_j].item())
    next_states = [0, 1, 2]
    next_states.remove(current)
    theory[target_i, target_j] = next_states[torch.randint(0, len(next_states), (1,)).item()]
    return True


def corrupt_theory(theory: torch.Tensor, max_steps: int | None = None) -> list[torch.Tensor]:
    trajectory = [theory.clone()]
    current = theory.clone()

    if max_steps is None:
        max_steps = current.size(0) * current.size(1)

    for _ in range(max_steps):
        active_cols = [j for j in range(current.size(1)) if (current[:, j] != 0).any()]
        if not active_cols: break
        muc_cols = set(get_muc_columns(current))
        mutable_cols = [j for j in active_cols if j not in muc_cols]
        if not mutable_cols: break

        model = get_satisfying_model(current, mutable_cols)
        before = current.clone()

        if model is None:
            changed = _random_mutation(current, mutable_cols)
        else:
            target_j = mutable_cols[torch.randint(0, len(mutable_cols), (1,)).item()]
            _mutate_clause_against_model(current, target_j, model)
            changed = not torch.equal(before, current)

        if not changed: continue
        trajectory.append(current.clone())

        post_muc = set(get_muc_columns(current))
        if post_muc and not [j for j in active_cols if j not in post_muc]: break

    safety_steps = 0
    max_safety_steps = current.size(0) * current.size(1) * 2
    while safety_steps < max_safety_steps:
        if not get_muc_columns(current):
            active_cols = [j for j in range(current.size(1)) if (current[:, j] != 0).any()]
            if not active_cols: break
            muc_cols = set(get_muc_columns(current))
            mutable_cols = [j for j in active_cols if j not in muc_cols]
            if not mutable_cols: break
            model = get_satisfying_model(current, mutable_cols)
            if model is None:
                _random_mutation(current, mutable_cols)
            else:
                target_j = mutable_cols[torch.randint(0, len(mutable_cols), (1,)).item()]
                _mutate_clause_against_model(current, target_j, model)
            trajectory.append(current.clone())
            safety_steps += 1
        else:
            break

    return trajectory


# ==========================================
# Curriculum + Split Builder
# ==========================================
def get_stage_bounds(stages: list[tuple[int, int]]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    for i, (_, stage_max) in enumerate(stages):
        min_val = 1 if i == 0 else stages[i - 1][1]
        bounds.append((min_val, stage_max))
    return bounds


def compute_curriculum_distribution(total_samples: int) -> list[int]:
    stage_durations = []
    for i in range(len(CURRICULUM_N_STAGES)):
        start = CURRICULUM_N_STAGES[i][0]
        end = CURRICULUM_N_STAGES[i + 1][0] if i + 1 < len(CURRICULUM_N_STAGES) else EPOCHS + 1
        stage_durations.append(end - start)

    total_epochs = sum(stage_durations)
    distribution = [int(total_samples * (d / total_epochs)) for d in stage_durations]
    distribution[-1] += total_samples - sum(distribution)
    return distribution


def _pad_theory(state: torch.Tensor, pad_n: int, pad_m: int) -> torch.Tensor:
    padded = torch.zeros((pad_n, pad_m), dtype=torch.long)
    padded[:state.size(0), :state.size(1)] = state
    return padded


def _stage_plan(num_samples: int, follow_curriculum: bool) -> list[tuple[int, int, int, int, int]]:
    if follow_curriculum:
        n_bounds = get_stage_bounds(CURRICULUM_N_STAGES)
        m_bounds = get_stage_bounds(CURRICULUM_M_STAGES)
        stage_sizes = compute_curriculum_distribution(num_samples)
    else:
        n_bounds = [(1, N_LITERALS)]
        m_bounds = [(1, M_CLAUSES)]
        stage_sizes = [num_samples]

    plan: list[tuple[int, int, int, int, int]] = []
    for stage_idx, count in enumerate(stage_sizes):
        if count <= 0: continue
        n_min, n_max = n_bounds[stage_idx]
        m_min, m_max = m_bounds[stage_idx]
        plan.append((n_min, n_max, m_min, m_max, count))
    return plan


def _sample_signature(clean: torch.Tensor) -> str:
    arr = clean.contiguous().cpu().numpy().tobytes()
    shape = str(tuple(clean.shape)).encode("ascii")
    return hashlib.sha1(arr + shape).hexdigest()


def _generate_sample(n_min: int, n_max: int, m_min: int, m_max: int, pad_n: int, pad_m: int) -> tuple[
    dict[str, torch.Tensor | int], str]:
    clean = generate_consistent_theory(
        N=pad_n,
        max_clauses=m_max,
        active_N=n_max,
        min_active_N=n_min,
        min_clauses=m_min,
    )

    corruption_steps = clean.size(0) * clean.size(1)
    trajectory = corrupt_theory(clean, max_steps=corruption_steps)
    padded_traj = [_pad_theory(state, pad_n, pad_m) for state in trajectory]

    entry: dict[str, torch.Tensor | int] = {
        "clean": padded_traj[0],
        "corrupted": padded_traj[-1],
        "trajectory": torch.stack(padded_traj),
        "steps": len(padded_traj) - 1,
        "original_n": n_max,
        "original_m": clean.size(1),
    }
    return entry, _sample_signature(clean)


def _build_splits_from_plans(
        plans: dict[str, list[tuple[int, int, int, int, int]]],
        pad_n: int,
        pad_m: int,
        max_workers: int | None = None,
) -> dict[str, list[dict[str, torch.Tensor | int]]]:
    datasets: dict[str, list[dict[str, torch.Tensor | int]]] = {name: [] for name in plans.keys()}

    queue: list[tuple[str, int, int, int, int]] = []
    for split_name, stage_plan in plans.items():
        print(f"Planning {split_name.upper()} split...")
        for stage_idx, (n_min, n_max, m_min, m_max, count) in enumerate(stage_plan, start=1):
            print(f"  -> N:[{n_min},{n_max}] M:[{m_min},{m_max}] | {count} samples")
            queue.extend([(split_name, n_min, n_max, m_min, m_max)] * count)

    random.shuffle(queue)
    resolved_workers = max_workers if max_workers is not None else min(4, os.cpu_count() or 1)
    print(f"\nGenerating {len(queue)} samples using {resolved_workers} CPU workers...")

    seen_signatures: set[str] = set()
    task_queue: list[tuple[str, int, int, int, int]] = list(queue)
    max_in_flight = max(4, resolved_workers * 4)

    with tqdm(total=len(queue), desc="Generating Samples") as pbar:
        while task_queue:
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=resolved_workers) as executor:
                    in_flight: dict[concurrent.futures.Future, tuple[str, int, int, int, int]] = {}

                    def submit_one(params: tuple[str, int, int, int, int]) -> None:
                        split_name, n_min, n_max, m_min, m_max = params
                        fut = executor.submit(_generate_sample, n_min, n_max, m_min, m_max, pad_n, pad_m)
                        in_flight[fut] = params

                    while task_queue and len(in_flight) < max_in_flight:
                        submit_one(task_queue.pop())

                    while in_flight:
                        done, _ = concurrent.futures.wait(
                            in_flight.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                        )

                        for fut in done:
                            params = in_flight.pop(fut)
                            split_name, n_min, n_max, m_min, m_max = params
                            try:
                                entry, signature = fut.result()
                                if signature in seen_signatures:
                                    task_queue.append(params)
                                else:
                                    seen_signatures.add(signature)
                                    datasets[split_name].append(entry)
                                    pbar.update(1)
                            except BrokenProcessPool:
                                task_queue.append(params)
                                task_queue.extend(in_flight.values())
                                in_flight.clear()
                                raise
                            except Exception as exc:
                                print(f"\nWorker generated an exception: {exc}")
                                task_queue.append(params)

                            while task_queue and len(in_flight) < max_in_flight:
                                submit_one(task_queue.pop())
            except BrokenProcessPool:
                print("\nProcess pool crashed; restarting workers...")

    return datasets


def build_all_splits_parallel(max_workers: int | None = None) -> dict[str, list[dict[str, torch.Tensor | int]]]:
    targets = {
        "train": (NUM_TRAIN, True),
        "val": (NUM_VAL, True),
        "test": (NUM_TEST, False),
    }
    plans = {name: _stage_plan(num, follow) for name, (num, follow) in targets.items()}
    return _build_splits_from_plans(plans=plans, pad_n=N_LITERALS, pad_m=M_CLAUSES, max_workers=max_workers)


def build_finetune_biggest_with_trajectories(
        total_samples: int,
        train_ratio: float = 0.8,
        max_workers: int | None = None,
) -> dict[str, list[dict[str, torch.Tensor | int]]]:
    train_count = int(total_samples * train_ratio)
    val_count = total_samples - train_count

    plans = {
        "finetune_train": [(N_LITERALS, N_LITERALS, M_CLAUSES, M_CLAUSES, train_count)],
        "finetune_val": [(N_LITERALS, N_LITERALS, M_CLAUSES, M_CLAUSES, val_count)],
    }
    return _build_splits_from_plans(plans=plans, pad_n=N_LITERALS, pad_m=M_CLAUSES, max_workers=max_workers)


def build_hyperparam_biggest_with_trajectories(
        total_samples: int,
        train_ratio: float = 0.8,
        max_workers: int | None = None,
) -> dict[str, list[dict[str, torch.Tensor | int]]]:
    train_count = int(total_samples * train_ratio)
    val_count = total_samples - train_count

    plans = {
        "hyperparam_train": [(N_LITERALS, N_LITERALS, M_CLAUSES, M_CLAUSES, train_count)],
        "hyperparam_val": [(N_LITERALS, N_LITERALS, M_CLAUSES, M_CLAUSES, val_count)],
    }
    return _build_splits_from_plans(plans=plans, pad_n=N_LITERALS, pad_m=M_CLAUSES, max_workers=max_workers)


def build_test_biggest_with_trajectories(
        total_samples: int,
        max_workers: int | None = None,
) -> dict[str, list[dict[str, torch.Tensor | int]]]:
    plans = {
        "test": [(N_LITERALS, N_LITERALS, M_CLAUSES, M_CLAUSES, total_samples)],
    }
    return _build_splits_from_plans(plans=plans, pad_n=N_LITERALS, pad_m=M_CLAUSES, max_workers=max_workers)


def build_generalization_dataset(
        samples_per_interval: int,
        num_intervals: int = 10,
        max_workers: int | None = None
) -> None:
    """Generates an extrapolation grid and saves one .pt file per (N, M) pair."""
    n_target = N_LITERALS * 2
    m_target = M_CLAUSES * 2

    n_step = (n_target - N_LITERALS) / num_intervals
    m_step = (m_target - M_CLAUSES) / num_intervals

    n_values = sorted({int(round(N_LITERALS + n_step * i)) for i in range(1, num_intervals + 1)})
    m_values = sorted({int(round(M_CLAUSES + m_step * i)) for i in range(1, num_intervals + 1)})

    combinations: list[tuple[int, int]] = [(n, m) for n in n_values for m in m_values]
    pending: list[tuple[int, int]] = []
    skipped = 0

    for curr_n, curr_m in combinations:
        save_path = OUTPUT_DIR / f"generalization_N{curr_n}_M{curr_m}.pt"
        if save_path.exists():
            skipped += 1
            print(f"Skipping existing file: {save_path.name}")
            continue
        pending.append((curr_n, curr_m))

    print(f"\nGeneralization grid prepared: {len(n_values)} x {len(m_values)} = {len(combinations)} combinations")
    print(f"Already present: {skipped} | To generate: {len(pending)}")

    for idx, (curr_n, curr_m) in enumerate(pending, start=1):

        print(f"\n{'=' * 50}")
        print(f"--- Generating Combo {idx}/{len(pending)}: N={curr_n}, M={curr_m} ---")
        print(f"{'=' * 50}")

        # Generate one fixed-size chunk for this (N, M) combination.
        plan = {"gen_chunk": [(curr_n, curr_n, curr_m, curr_m, samples_per_interval)]}

        chunk_data_dict = _build_splits_from_plans(
            plans=plan,
            pad_n=n_target,
            pad_m=m_target,
            max_workers=max_workers
        )

        chunk_list = chunk_data_dict["gen_chunk"]

        save_path = OUTPUT_DIR / f"generalization_N{curr_n}_M{curr_m}.pt"
        torch.save(chunk_list, save_path)
        print(f"✅ Saved {len(chunk_list)} samples to {save_path.name}")

        del chunk_list
        del chunk_data_dict

    if not pending:
        print("All generalization combinations already exist. Nothing to generate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate offline datasets in parallel.")
    parser.add_argument(
        "--mode",
        choices=["default", "finetune_biggest", "hyperparam_biggest", "test_biggest", "generalization"],
        default="default",
        help="default: train/val/test curriculum dataset; finetune_biggest: max-size finetune; hyperparam_biggest: max-size hyperparam; test_biggest: max-size test; generalization: multi-interval scaled dataset.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
        help="Process workers to use (default: min(4, cpu_count)).",
    )
    parser.add_argument(
        "--finetune-total",
        type=int,
        default=12800,
        help="Total finetune samples when --mode finetune_biggest (default: 12800).",
    )
    parser.add_argument(
        "--hyperparam-total",
        type=int,
        default=12800,
        help="Total hyperparam samples when --mode hyperparam_biggest (default: 12800).",
    )
    parser.add_argument(
        "--test-total",
        type=int,
        default=NUM_TEST,
        help="Total test samples when --mode test_biggest (default: NUM_TEST).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio for finetune_biggest and hyperparam_biggest modes (default: 0.8).",
    )
    parser.add_argument(
        "--gen-total",
        type=int,
        default=300,
        help="Samples PER generalization interval when --mode generalization (default: 100)."
    )
    parser.add_argument(
        "--gen-intervals",
        type=int,
        default=10,
        help="Number of generalization intervals to scale up to 2x (default: 10)."
    )
    args = parser.parse_args()

    print(f"Starting Offline Dataset Generation (Mode: {args.mode})...")

    if args.mode == "default":
        split_data = build_all_splits_parallel(max_workers=args.max_workers)
        train_data = split_data["train"]
        val_data = split_data["val"]
        test_data = split_data["test"]
        print(f"\nFinal split sizes | train={len(train_data)} val={len(val_data)} test={len(test_data)}")
        print("\nSaving datasets to disk...")
        torch.save(train_data, OUTPUT_DIR / "train.pt")
        torch.save(val_data, OUTPUT_DIR / "val.pt")
        torch.save(test_data, OUTPUT_DIR / "test.pt")

    elif args.mode == "finetune_biggest":
        split_data = build_finetune_biggest_with_trajectories(total_samples=args.finetune_total,
                                                              train_ratio=args.train_ratio,
                                                              max_workers=args.max_workers)
        finetune_train = split_data["finetune_train"]
        finetune_val = split_data["finetune_val"]
        finetune_all = finetune_train + finetune_val
        print(
            f"\nFinal split sizes | finetune_train={len(finetune_train)} finetune_val={len(finetune_val)} total={len(finetune_all)}")
        print("\nSaving datasets to disk...")
        torch.save(finetune_train, OUTPUT_DIR / "finetune_train.pt")
        torch.save(finetune_val, OUTPUT_DIR / "finetune_val.pt")
        torch.save(finetune_all, OUTPUT_DIR / "finetune.pt")

    elif args.mode == "hyperparam_biggest":
        split_data = build_hyperparam_biggest_with_trajectories(total_samples=args.hyperparam_total,
                                                                train_ratio=args.train_ratio,
                                                                max_workers=args.max_workers)
        hyperparam_train = split_data["hyperparam_train"]
        hyperparam_val = split_data["hyperparam_val"]
        hyperparam_all = hyperparam_train + hyperparam_val
        print(
            f"\nFinal split sizes | hyperparam_train={len(hyperparam_train)} hyperparam_val={len(hyperparam_val)} total={len(hyperparam_all)}")
        print("\nSaving datasets to disk...")
        torch.save(hyperparam_train, OUTPUT_DIR / "hyperparam_train.pt")
        torch.save(hyperparam_val, OUTPUT_DIR / "hyperparam_val.pt")
        torch.save(hyperparam_all, OUTPUT_DIR / "hyperparam.pt")

    elif args.mode == "test_biggest":
        split_data = build_test_biggest_with_trajectories(total_samples=args.test_total, max_workers=args.max_workers)
        test_data = split_data["test"]
        print(f"\nFinal split sizes | test={len(test_data)}")
        print("\nSaving datasets to disk...")
        torch.save(test_data, OUTPUT_DIR / "test.pt")

    elif args.mode == "generalization":
        # It handles its own saving directly now
        build_generalization_dataset(
            samples_per_interval=args.gen_total,
            num_intervals=args.gen_intervals,
            max_workers=args.max_workers
        )
        print("\nAll generalization intervals saved successfully to disk.")

    print(f"Success! Datasets saved to {OUTPUT_DIR.resolve()}")