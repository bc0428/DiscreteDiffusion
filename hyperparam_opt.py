"""
tune_rl_hyperparams.py
======================
Optimizes RL reward parameters and learning rate for the D3PM Theory Denoiser
using Multi-Objective Bayesian Optimization (Optuna).

Target Metrics:
1. Maximize Consistency Success Rate
2. Minimize Total Change Percentage
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

sys.path.insert(0, str(Path(__file__).parent))
from main import (
    TheoryDenoiserNet,
    D3PMForwardCorruption,
    get_dataloaders,
    is_consistent,
    resolve_curriculum_max_clauses
)

# ==========================================
# Tuning Configuration
# ==========================================
CHECKPOINT_PATH = "outputs/theory_denoiser_base.pt"
EPOCHS_PER_TRIAL = 20
NUM_ROLLOUTS_PER_EPOCH = 16
N_TRIALS = 50


def run_tuning_rollouts(model, corrupt, dataloader, optimizer, device, params, update_model=True, clause_mask_batch=None):
    if update_model:
        model.train()
    else:
        model.eval()

    if clause_mask_batch is not None:
        clause_mask = clause_mask_batch
    else:
        try:
            _, clause_mask = next(iter(dataloader))
        except StopIteration:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    batch_size = clause_mask.size(0)
    clause_mask = clause_mask.to(device)
    valid_mask_float = clause_mask.unsqueeze(1).float()

    x_t = torch.randint(0, model.num_classes, (batch_size, model.N, clause_mask.size(1)), device=device)
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
        for t_step in reversed(range(1, model.num_timesteps + 1)):
            if not active_mask.any():
                break

            t_tensor = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
            saved_transitions.append({
                'x_t': x_t.clone(),
                't_tensor': t_tensor.clone(),
                'active_mask': active_mask.clone()
            })

            logits = model(x_t, t_tensor, clause_mask=clause_mask)
            dist = torch.distributions.Categorical(logits=logits)
            x_0_pred = dist.sample()
            x_0_pred = x_0_pred.masked_fill(~clause_mask.unsqueeze(1), 0)
            saved_transitions[-1]['x_0_pred'] = x_0_pred.clone()

            if t_step > 1:
                t_minus_1 = torch.full((batch_size,), t_step - 1, device=device, dtype=torch.long)
                x_t_minus_1 = corrupt.q_sample(x_0_pred, t_minus_1, clause_mask=clause_mask)
            else:
                x_t_minus_1 = x_0_pred

            for b in range(batch_size):
                if not active_mask[b]:
                    continue

                # Continuously update x_final to the most recent prediction
                x_final[b] = x_0_pred[b].clone()

                r = 0.0
                active_theory = x_0_pred[b, :, clause_mask[b]].detach().cpu()

                # Terminal Reward Logic
                if active_theory.numel() > 0 and is_consistent(active_theory):
                    # 1. Base Payout & Speed Bonus (Normalized to 0.0 - 1.0)
                    time_saved_pct = (t_step - 1) / model.num_timesteps
                    gross_reward = params['massive_reward'] + (params['early_finish_bonus'] * time_saved_pct)

                    # 2. Terminal Edit Tax
                    total_changes = ((x_initial[b] != x_0_pred[b]) * valid_mask_float[b]).sum().item()
                    edit_tax = params['change_penalty'] * total_changes

                    # 3. Final Net Reward
                    r = gross_reward - edit_tax

                    active_mask[b] = False
                    hits += 1

                traj_rewards[b].append(r)

            x_t = x_t_minus_1

    # ── METRIC CALCULATION: Total and Empty Changes ──
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


def objective(trial):
    params = {
        'lr': trial.suggest_float('lr', 1e-6, 5e-5, log=True),
        'change_penalty': trial.suggest_float('change_penalty', 0.5, 5.0),
        'massive_reward': trial.suggest_float('massive_reward', 50.0, 200.0),
        'early_finish_bonus': trial.suggest_float('early_finish_bonus', 0.0, 50.0),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = TheoryDenoiserNet(N=cfg["N_LITERALS"], M=cfg["M_CLAUSES"], num_classes=cfg["K_STATES"],
                              num_timesteps=cfg["NUM_TIMESTEPS"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    corrupt = D3PMForwardCorruption(num_classes=cfg["K_STATES"], num_timesteps=cfg["NUM_TIMESTEPS"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    final_max_clauses = resolve_curriculum_max_clauses(999, cfg["CURRICULUM_STAGES"], cfg["M_CLAUSES"])
    train_loader, test_loader, _ = get_dataloaders(
        num_samples=NUM_ROLLOUTS_PER_EPOCH * cfg["BATCH_SIZE"],
        N=cfg["N_LITERALS"], max_clauses=final_max_clauses,
        batch_size=cfg["BATCH_SIZE"], train_ratio=0.8
    )

    print(f"\n--- Starting Trial {trial.number} ---")
    for epoch in range(1, EPOCHS_PER_TRIAL + 1):
        for _ in range(NUM_ROLLOUTS_PER_EPOCH):
            run_tuning_rollouts(model, corrupt, train_loader, optimizer, device, params)
        print(f"Epoch {epoch}/{EPOCHS_PER_TRIAL} | TRAINING ONLY")

    test_sr_list = []
    test_chg_list = []
    test_e_bef_list = []
    test_e_aft_list = []
    test_net_list = []

    with torch.no_grad():
        for _, clause_mask in test_loader:
            _, sr, _, chg, e_bef, e_aft, bsz = run_tuning_rollouts(
                model,
                corrupt,
                dataloader=None,
                optimizer=None,
                device=device,
                params=params,
                update_model=False,
                clause_mask_batch=clause_mask,
            )
            if bsz == 0:
                continue
            test_sr_list.append(sr)
            test_chg_list.append(chg)
            test_e_bef_list.append(e_bef)
            test_e_aft_list.append(e_aft)
            test_net_list.append(e_aft - e_bef)

    def mean_std(values):
        if not values:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        if len(values) == 1:
            return mean, 0.0
        var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return mean, math.sqrt(var)

    final_success_rate, final_sr_std = mean_std(test_sr_list)
    final_chg, final_chg_std = mean_std(test_chg_list)
    final_e_bef, final_e_bef_std = mean_std(test_e_bef_list)
    final_e_aft, final_e_aft_std = mean_std(test_e_aft_list)
    final_net_change, final_net_std = mean_std(test_net_list)

    abs_empty_change = abs(final_net_change)

    print(
        f"Final TEST | SR: {final_success_rate:.1f} +/- {final_sr_std:.1f}% | "
        f"Total Edit: {final_chg:.1f} +/- {final_chg_std:.1f}% | "
        f"Empty: {final_e_bef:.1f} +/- {final_e_bef_std:.1f}% -> {final_e_aft:.1f} +/- {final_e_aft_std:.1f}% | "
        f"Net: {final_net_change:+.1f} +/- {final_net_std:.1f}%"
    )

    trial.set_user_attr("Success Rate Std %", final_sr_std)
    trial.set_user_attr("Total Change %", final_chg)
    trial.set_user_attr("Total Change Std %", final_chg_std)
    trial.set_user_attr("Empty % Before", final_e_bef)
    trial.set_user_attr("Empty % Before Std", final_e_bef_std)
    trial.set_user_attr("Empty % After", final_e_aft)
    trial.set_user_attr("Empty % After Std", final_e_aft_std)
    trial.set_user_attr("Net Empty Change %", final_net_change)
    trial.set_user_attr("Net Empty Change Std %", final_net_std)

    del model
    del optimizer
    torch.cuda.empty_cache()

    # Optuna will: Maximize SR, Minimize Total Change, Minimize Absolute Empty Change
    return final_success_rate, final_chg, abs_empty_change


def main():
    print("Starting Multi-Objective Bayesian Hyperparameter Optimization...")

    # Multi-objective study: directions map directly to the returned tuple
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"]
    )

    study.optimize(objective, n_trials=N_TRIALS)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"hyperparam_trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    rows = []
    for tr in study.trials:
        # For multi-objective, tr.values is a list rather than a single tr.value
        sr_val = tr.values[0] if tr.values else None

        rows.append({
            "trial": tr.number,
            "state": str(tr.state),
            "value": sr_val, # Keeping the primary metric here for backward compatibility with your CSV
            "lr": tr.params.get("lr"),
            "change_penalty": tr.params.get("change_penalty"),
            "massive_reward": tr.params.get("massive_reward"),
            "early_finish_bonus": tr.params.get("early_finish_bonus"),
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
            "success_rate_std_pct",
            "total_change_pct", "total_change_std_pct",
            "empty_before_pct", "empty_before_std_pct",
            "empty_after_pct", "empty_after_std_pct",
            "net_empty_change_pct", "net_empty_change_std_pct"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 50)
    print("OPTIMIZATION FINISHED")
    print(f"\nSaved full trial table to: {csv_path}")

    print("\n--- Pareto Front (Best Trade-off Trials) ---")
    best_trials = study.best_trials

    if not best_trials:
        print("No Pareto front found.")
        return

    for i, t in enumerate(best_trials):
        print(f"\nPareto Option #{i+1} (Trial {t.number}):")
        print(f"  Success Rate:  {t.values[0]:.2f}%")
        print(f"  Total Change:  {t.values[1]:.2f}%")
        print(f"  Net Empty Chg: {t.values[2]:.2f}% (Absolute)")
        print("  Parameters:")
        for key, value in t.params.items():
            print(f"    {key}: {value:.5f}" if isinstance(value, float) else f"    {key}: {value}")


if __name__ == "__main__":
    main()