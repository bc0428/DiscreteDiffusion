"""
tune_rl_hyperparams.py
======================
Optimizes RL reward parameters and learning rate for the D3PM Theory Denoiser
using Multi-Objective Bayesian Optimization (Optuna).

Target Metrics:
1. Maximize Consistency Success Rate
2. Minimize Total Change Percentage (Minimal Edit Distance from Corrupted State)
3. Minimize Absolute Net Empty Change
"""

import os
import csv
import math
from datetime import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path
import optuna
import torch
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from main import (
    TheoryDenoiserNet,
    D3PMForwardCorruption,
    is_consistent,
)

# ==========================================
# Tuning Configuration
# ==========================================
CHECKPOINT_PATH = "outputs/theory_denoiser_base.pt"
EPOCHS_PER_TRIAL = 150
NUM_ROLLOUTS_PER_EPOCH = 16
N_TRIALS = 50
START_DENOISE_TRAJ_PCT_MIN = 0.3  # Minimum trajectory progress to start denoising from (30%)
START_DENOISE_TRAJ_PCT_MAX = 0.8  # Maximum trajectory progress to start denoising from (80%)
DATASET_DIR = Path("dataset")


def resolve_base_checkpoint() -> Path:
    preferred = Path(CHECKPOINT_PATH)
    if preferred.exists():
        return preferred

    candidates = sorted(Path("outputs").glob("theory_denoiser_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No base checkpoint found. Expected '{CHECKPOINT_PATH}' or at least one outputs/theory_denoiser_*.pt"
    )


class HyperparamDataset(Dataset):
    """Returns clean target + trajectory-selected start state for tuning rollouts."""
    def __init__(self, data: list[dict], num_timesteps: int, start_traj_pct_min: float, start_traj_pct_max: float):
        self.data = data
        self.num_timesteps = int(max(2, num_timesteps))
        self.start_traj_pct_min = max(0.0, min(1.0, float(start_traj_pct_min)))
        self.start_traj_pct_max = max(0.0, min(1.0, float(start_traj_pct_max)))
        if self.start_traj_pct_min > self.start_traj_pct_max:
            self.start_traj_pct_min, self.start_traj_pct_max = self.start_traj_pct_max, self.start_traj_pct_min
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        clean_theory = entry["clean"].clone().long()
        corrupted = entry.get("corrupted", clean_theory).clone().long()
        traj = entry.get("trajectory", None)

        if isinstance(traj, torch.Tensor) and traj.ndim == 3 and traj.size(0) > 1:
            max_idx = traj.size(0) - 1
            sampled_pct = self.start_traj_pct_min + torch.rand(1).item() * (self.start_traj_pct_max - self.start_traj_pct_min)
            idx_t = int(round(sampled_pct * max_idx))
            idx_t = max(1, min(max_idx, idx_t))
            start_state = traj[idx_t].clone().long()
            progress = idx_t / max(1, max_idx)
        else:
            start_state = corrupted
            progress = 1.0

        mapped_t = int(round(progress * (self.num_timesteps - 1)))
        start_t = max(1, min(self.num_timesteps - 1, mapped_t))

        clause_mask = (clean_theory.sum(dim=0) != 0).bool()
        return clean_theory, start_state, clause_mask, start_t


def theory_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
    """Collate function for HyperparamDataset."""
    batch_size = len(batch)
    N = batch[0][0].size(0)
    max_m = max(clean.size(1) for clean, _, _, _ in batch)

    x_0 = torch.zeros((batch_size, N, max_m), dtype=torch.long)
    x_start = torch.zeros((batch_size, N, max_m), dtype=torch.long)
    clause_mask = torch.zeros((batch_size, max_m), dtype=torch.bool)
    start_t = torch.zeros((batch_size,), dtype=torch.long)

    for b, (clean, start_state, mask, t_val) in enumerate(batch):
        m_i = clean.size(1)
        x_0[b, :, :m_i] = clean
        x_start[b, :, :m_i] = start_state[:, :m_i]
        clause_mask[b, :m_i] = mask
        start_t[b] = int(t_val)

    return x_0, x_start, clause_mask, start_t


# --- ADD THIS TO THE GLOBAL SCOPE (Above the function) ---
TRAIN_DATA_CACHE = None
VAL_DATA_CACHE = None

def load_hyperparam_dataloaders(batch_size: int, num_timesteps: int, start_pct_min: float = None,
                                start_pct_max: float = None):
    global TRAIN_DATA_CACHE, VAL_DATA_CACHE

    if start_pct_min is None:
        start_pct_min = START_DENOISE_TRAJ_PCT_MIN
    if start_pct_max is None:
        start_pct_max = START_DENOISE_TRAJ_PCT_MAX

    # ONLY load from disk if it hasn't been loaded yet
    if TRAIN_DATA_CACHE is None:
        print("Loading massive train dataset into RAM (This happens ONLY ONCE)...", flush=True)
        TRAIN_DATA_CACHE = torch.load(DATASET_DIR / "hyperparam_train.pt", weights_only=False)
    if VAL_DATA_CACHE is None:
        print("Loading massive val dataset into RAM (This happens ONLY ONCE)...", flush=True)
        VAL_DATA_CACHE = torch.load(DATASET_DIR / "hyperparam_val.pt", weights_only=False)

    train_dataset = HyperparamDataset(TRAIN_DATA_CACHE, num_timesteps=num_timesteps, start_traj_pct_min=start_pct_min,
                                      start_traj_pct_max=start_pct_max)
    val_dataset = HyperparamDataset(VAL_DATA_CACHE, num_timesteps=num_timesteps, start_traj_pct_min=start_pct_min,
                                    start_traj_pct_max=start_pct_max)

    # SET SHUFFLE TO FALSE: Load sequentially since the dataset is already randomized
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=theory_collate_fn,
                              num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=theory_collate_fn,
                            num_workers=0)

    return train_loader, val_loader

def run_tuning_rollouts(model, corrupt, optimizer, device, params, x_0_batch, x_start_batch, clause_mask_batch, start_t_batch, update_model=True):
    if update_model:
        model.train()
    else:
        model.eval()

    batch_size = x_0_batch.size(0)
    clause_mask = clause_mask_batch.to(device)
    valid_mask_float = clause_mask.unsqueeze(1).float()
    start_t_per_sample = start_t_batch.to(device).long()
    max_start_t = int(start_t_per_sample.max().item())

    # Start from offline MUC trajectory states instead of fresh q-sampled noise.
    x_t = x_start_batch.to(device)
    x_t = x_t.masked_fill(~clause_mask.unsqueeze(1), 0)

    # Track states for our metrics and terminal rewards
    x_initial = x_t.clone()
    x_final = torch.zeros_like(x_t)

    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    saved_transitions = []
    traj_rewards = [[] for _ in range(batch_size)]
    hits = 0

    # ── PHASE 1: NO-GRADIENT ROLLOUT ──
    with torch.no_grad():
        for t_step in reversed(range(1, max_start_t + 1)):
            if not active_mask.any():
                break

            step_active = active_mask & (start_t_per_sample >= t_step)
            if not step_active.any():
                continue

            t_tensor = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
            saved_transitions.append({
                'x_t': x_t.clone(),
                't_tensor': t_tensor.clone(),
                'active_mask': step_active.clone()
            })

            logits = model(x_t, t_tensor, clause_mask=clause_mask)
            dist = torch.distributions.Categorical(logits=logits)
            sampled = dist.sample().masked_fill(~clause_mask.unsqueeze(1), 0)
            x_0_pred = torch.where(step_active.view(-1, 1, 1), sampled, x_t)
            saved_transitions[-1]['x_0_pred'] = x_0_pred.clone()

            if t_step > 1:
                t_minus_1 = torch.full((batch_size,), t_step - 1, device=device, dtype=torch.long)
                x_t_minus_1 = corrupt.q_sample(x_0_pred, t_minus_1, clause_mask=clause_mask)
                x_t_minus_1 = torch.where(step_active.view(-1, 1, 1), x_t_minus_1, x_t)
            else:
                x_t_minus_1 = x_0_pred

            for b in range(batch_size):
                if not step_active[b]:
                    continue

                x_final[b] = x_0_pred[b].clone()
                r = 0.0
                active_theory = x_0_pred[b, :, clause_mask[b]].detach().cpu()

                # Terminal Reward Logic
                if active_theory.numel() > 0 and is_consistent(active_theory):
                    # Use per-sample starting step when starts are sampled from trajectories.
                    start_t_b = max(1, int(start_t_per_sample[b].item()))
                    time_saved_pct = (t_step - 1) / start_t_b
                    gross_reward = params['massive_reward'] + (params['early_finish_bonus'] * time_saved_pct)

                    valid_cells_b = valid_mask_float[b].sum().item() * model.N
                    total_changes = ((x_initial[b] != x_0_pred[b]) * valid_mask_float[b]).sum().item()

                    # 1. Edit Tax (NORMALIZED TO DECIMAL)
                    change_ratio = total_changes / max(1.0, valid_cells_b)
                    # Exponentiate the decimal, then scale up to the reward space
                    edit_tax = params['change_penalty'] * (change_ratio ** 1.5) * 100.0

                    # 2. Empty Penalty (NORMALIZED TO DECIMAL)
                    empty_before = ((x_initial[b] == 0) * valid_mask_float[b]).sum().item()
                    empty_after = ((x_0_pred[b] == 0) * valid_mask_float[b]).sum().item()

                    net_empty_ratio = (empty_after - empty_before) / max(1.0, valid_cells_b)
                    # Only penalize positive changes (erasure)
                    net_empty_ratio = max(0.0, net_empty_ratio)
                    empty_tax = params['empty_penalty'] * net_empty_ratio * 100.0

                    r = gross_reward - edit_tax - empty_tax

                    active_mask[b] = False
                    hits += 1
                else:
                    # 3. Critical Failure Penalty (Only on final step)
                    if t_step == 1:
                        r = -params['massive_reward']
                    else:
                        r = 0.0

                traj_rewards[b].append(r)

            x_t = x_t_minus_1

    # ── METRIC CALCULATION ──
    valid_cells = (valid_mask_float.sum(dim=(1, 2)) * model.N).clamp_min(1.0)
    changed_cells = ((x_initial != x_final) * valid_mask_float).sum(dim=(1, 2))
    empty_before_cells = ((x_initial == 0) * valid_mask_float).sum(dim=(1, 2))
    empty_after_cells = ((x_final == 0) * valid_mask_float).sum(dim=(1, 2))

    total_change_pct = (changed_cells / valid_cells).mean().item() * 100.0
    empty_pct_before = (empty_before_cells / valid_cells).mean().item() * 100.0
    empty_pct_after = (empty_after_cells / valid_cells).mean().item() * 100.0

    # ── PHASE 2: CALCULATE RETURNS ──
    all_returns = []
    batch_returns_lists = []

    for b in range(batch_size):
        total_episode_return = sum(traj_rewards[b])
        returns = [total_episode_return for _ in traj_rewards[b]]
        batch_returns_lists.append(returns)
        if returns:
            all_returns.extend(returns)

    if not all_returns:
        return 0.0, 0.0, 0.0, total_change_pct, empty_pct_before, empty_pct_after, batch_size

    baseline = sum(all_returns) / len(all_returns)

    # ── PHASE 3: STEP-WISE BACKPROPAGATION ──
    if not update_model:
        return 0.0, (hits / batch_size) * 100, baseline, total_change_pct, empty_pct_before, empty_pct_after, batch_size

    optimizer.zero_grad()
    total_policy_loss_val = 0.0
    b_step_indices = [0 for _ in range(batch_size)]
    use_amp = device.type == "cuda"

    for trans in saved_transitions:
        active_curr = trans['active_mask']
        if not active_curr.any():
            continue

        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            logits = model(trans['x_t'], trans['t_tensor'], clause_mask=clause_mask)
            dist = torch.distributions.Categorical(logits=logits)
            step_log_probs = (dist.log_prob(trans['x_0_pred']) * valid_mask_float).sum(dim=(1, 2))

            step_loss = 0.0
            has_loss = False

            for b in range(batch_size):
                if active_curr[b]:
                    G_t = batch_returns_lists[b][b_step_indices[b]]
                    b_step_indices[b] += 1
                    advantage = G_t - baseline
                    step_loss = step_loss - (step_log_probs[b] * advantage) / batch_size
                    has_loss = True

        if has_loss:
            step_loss.backward()
            total_policy_loss_val += step_loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_policy_loss_val, (hits / batch_size) * 100, baseline, total_change_pct, empty_pct_before, empty_pct_after, batch_size

def get_infinite_batches(dataloader):
    """Yields batches sequentially and loops back to the start automatically."""
    while True:
        for batch in dataloader:
            yield batch

def objective(trial):
    params = {
        'lr': trial.suggest_float('lr', 1e-6, 8e-5, log=True),
        'change_penalty': trial.suggest_float('change_penalty', 0.01, 0.6, log=True),
        'empty_penalty': trial.suggest_float('empty_penalty', 4.0, 16.0, step=1.0),
        'massive_reward': trial.suggest_float('massive_reward', 120.0, 400.0, step=20.0),
        'early_finish_bonus': trial.suggest_float('early_finish_bonus', 0.0, 16.0, step=1.0),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(resolve_base_checkpoint(), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = TheoryDenoiserNet(N=cfg["N_LITERALS"], M=cfg["M_CLAUSES"], num_classes=cfg["K_STATES"],
                              num_timesteps=cfg["NUM_TIMESTEPS"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    corrupt = D3PMForwardCorruption(num_classes=cfg["K_STATES"], num_timesteps=cfg["NUM_TIMESTEPS"]).to(device)

    train_loader, val_loader = load_hyperparam_dataloaders(
        batch_size=cfg["BATCH_SIZE"],
        num_timesteps=cfg["NUM_TIMESTEPS"],
        start_pct_min=START_DENOISE_TRAJ_PCT_MIN,
        start_pct_max=START_DENOISE_TRAJ_PCT_MAX
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_TRIAL, eta_min=1e-6)

    # Calculate how many batches make up one "Epoch" based on your rollout target
    batches_per_epoch = max(1, NUM_ROLLOUTS_PER_EPOCH // cfg["BATCH_SIZE"])

    # Create the persistent sequential iterator
    train_iter = iter(get_infinite_batches(train_loader))

    print(f"\n--- Starting Trial {trial.number} ---")
    for epoch in range(1, EPOCHS_PER_TRIAL + 1):
        for _ in range(batches_per_epoch):
            x_0_batch, x_start_batch, clause_mask, start_t_batch = next(train_iter)
            run_tuning_rollouts(
                model, corrupt, optimizer, device, params,
                x_0_batch=x_0_batch, x_start_batch=x_start_batch, clause_mask_batch=clause_mask,
                start_t_batch=start_t_batch, update_model=True
            )

        scheduler.step()
        print(f"Epoch {epoch}/{EPOCHS_PER_TRIAL} | TRAINING ONLY")

    test_sr_list, test_chg_list, test_e_bef_list, test_e_aft_list, test_net_e_list = [], [], [], [], []

    with torch.no_grad():
        for x_0_batch, x_start_batch, clause_mask, start_t_batch in val_loader:
            _, sr, _, chg, e_bef, e_aft, bsz = run_tuning_rollouts(
                model, corrupt, optimizer=None, device=device, params=params,
                x_0_batch=x_0_batch, x_start_batch=x_start_batch, clause_mask_batch=clause_mask,
                start_t_batch=start_t_batch, update_model=False
            )
            if bsz == 0:
                continue
            test_sr_list.append(sr)
            test_chg_list.append(chg)
            test_e_bef_list.append(e_bef)
            test_e_aft_list.append(e_aft)
            test_net_e_list.append(e_aft - e_bef)

    def mean_std(values):
        if not values: return 0.0, 0.0
        mean = sum(values) / len(values)
        if len(values) == 1: return mean, 0.0
        var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return mean, math.sqrt(var)

    success_rate_mean, final_sr_std = mean_std(test_sr_list)
    chg_mean, chg_std = mean_std(test_chg_list)
    e_bef_mean, e_bef_std = mean_std(test_e_bef_list)
    e_aft_mean, e_aft_std = mean_std(test_e_aft_list)
    net_e_change_mean, net_e_change_std = mean_std(test_net_e_list)

    print(
        f"Final TEST | SR: {success_rate_mean:.1f} +/- {final_sr_std:.1f}% | "
        f"Total Edit: {chg_mean:.1f} +/- {chg_std:.1f}% | "
        f"Empty: {e_bef_mean:.1f} +/- {e_bef_std:.1f}% -> {e_aft_mean:.1f} +/- {e_aft_std:.1f}% | "
        f"Net: {net_e_change_mean:+.1f} +/- {net_e_change_std:.1f}%"
    )

    trial.set_user_attr("Success Rate Std %", final_sr_std)
    trial.set_user_attr("Total Change %", chg_mean)
    trial.set_user_attr("Total Change Std %", chg_std)
    trial.set_user_attr("Empty % Before", e_bef_mean)
    trial.set_user_attr("Empty % Before Std", e_bef_std)
    trial.set_user_attr("Empty % After", e_aft_mean)
    trial.set_user_attr("Empty % After Std", e_aft_std)
    trial.set_user_attr("Net Empty Change %", net_e_change_mean)
    trial.set_user_attr("Net Empty Change Std %", net_e_change_std)

    del model
    del optimizer
    torch.cuda.empty_cache()

    return success_rate_mean, chg_mean, max(0.0, net_e_change_mean)


def main():
    print("Starting Multi-Objective Bayesian Hyperparameter Optimization...")

    # 1. Create a database path in your outputs folder
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "optuna_d3pm_tuning.db"

    # 2. Tell Optuna to use SQLite and load past results if they exist
    study = optuna.create_study(
        study_name="d3pm_theory_denoiser",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        directions=["maximize", "minimize", "minimize"]
    )

    # 3. Run optimization
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=2)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"hyperparam_trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    rows = []
    for tr in study.trials:
        sr_val = tr.values[0] if tr.values else None
        rows.append({
            "trial": tr.number,
            "state": str(tr.state),
            "value": sr_val,
            "lr": tr.params.get("lr"),
            "change_penalty": tr.params.get("change_penalty"),
            "massive_reward": tr.params.get("massive_reward"),
            "early_finish_bonus": tr.params.get("early_finish_bonus"),
            "empty_penalty": tr.params.get("empty_penalty"),
            "success_rate_std_pct": tr.user_attrs.get("Success Rate Std %"),
            "total_change_pct": tr.user_attrs.get("Total Change %"),
            "total_change_std_pct": tr.user_attrs.get("Total Change Std %"),
            "empty_before_pct": tr.user_attrs.get("Empty % Before"),
            "empty_before_std_pct": tr.user_attrs.get("Empty % Before Std"),
            "empty_after_pct": tr.user_attrs.get("Empty % After"),
            "empty_after_std_pct": tr.user_attrs.get("Empty % After Std"),
            "net_empty_change_pct": tr.user_attrs.get("Net Empty Change %"),
            "net_empty_change_std_pct": tr.user_attrs.get("Net Empty Change Std %"),
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "trial", "state", "value", "lr", "change_penalty", "massive_reward", "early_finish_bonus",
            "empty_penalty", "success_rate_std_pct", "total_change_pct", "total_change_std_pct",
            "empty_before_pct", "empty_before_std_pct", "empty_after_pct", "empty_after_std_pct",
            "net_empty_change_pct", "net_empty_change_std_pct"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 50)
    print("OPTIMIZATION FINISHED")
    print(f"\nSaved full trial table to: {csv_path}")

    print("\n--- Pareto Front (Best Trade-off Trials) ---")
    if not study.best_trials:
        print("No Pareto front found.")
        return

    for i, t in enumerate(study.best_trials):
        print(f"\nPareto Option #{i+1} (Trial {t.number}):")
        print(f"  Success Rate:  {t.values[0]:.2f}%")
        print(f"  Total Change:  {t.values[1]:.2f}%")
        print(f"  Net Empty Chg: {t.values[2]:.2f}% (after relu)")
        print("  Parameters:")
        for key, value in t.params.items():
            print(f"    {key}: {value:.5f}" if isinstance(value, float) else f"    {key}: {value}")

if __name__ == "__main__":
    main()