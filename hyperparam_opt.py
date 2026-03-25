"""
tune_rl_hyperparams.py
======================
Optimizes RL reward parameters and learning rate for the D3PM Theory Denoiser
using Bayesian Optimization (Optuna).

Target Metric: Maximize the Consistency Success Rate.
Reports: Total edits, and empty cell percentages before and after revision.
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
EPOCHS_PER_TRIAL = 20  # Keep this low to iterate quickly
NUM_ROLLOUTS_PER_EPOCH = 16  # Reduced for faster trial evaluation
N_TRIALS = 50  # Total number of hyperparameter combinations to test


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

    # Track states for our new metrics
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

                r = 0
                active_theory = x_0_pred[b, :, clause_mask[b]].detach().cpu()

                if active_theory.numel() > 0 and is_consistent(active_theory):
                    r = params['massive_reward'] + (params['early_finish_bonus'] * (t_step - 1))
                    active_mask[b] = False
                    hits += 1

                changes = ((x_t[b] != x_t_minus_1[b]) * valid_mask_float[b]).sum().item()
                r += - (params['step_cost'] * changes)
                traj_rewards[b].append(r)

            x_t = x_t_minus_1

    # ── METRIC CALCULATION: Total and Empty Changes ──
    # Calculate the total number of valid cells for each theory in the batch
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
        # FIX: Ensure the undiscounted return applies to every step in the trajectory
        # so `b_step_indices` doesn't throw an IndexError during backprop
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
        'step_cost': 1.0,
        'massive_reward': trial.suggest_float('massive_reward', 50.0, 300.0, step=50.0),
        'early_finish_bonus': trial.suggest_float('early_finish_bonus', 1.0, 10.0, step=3.0),
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
        # Train on train split for this hyperparameter config.
        for _ in range(NUM_ROLLOUTS_PER_EPOCH):
            run_tuning_rollouts(model, corrupt, train_loader, optimizer, device, params)

        print(f"Epoch {epoch}/{EPOCHS_PER_TRIAL} | TRAINING ONLY")

    # Evaluate once on the entire test split after full training for this trial.
    # Build metric lists first, then compute mean/std from the same lists.
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

    print(
        f"Final TEST | SR: {final_success_rate:.1f} +/- {final_sr_std:.1f}% | "
        f"Total Edit: {final_chg:.1f} +/- {final_chg_std:.1f}% | "
        f"Empty: {final_e_bef:.1f} +/- {final_e_bef_std:.1f}% -> {final_e_aft:.1f} +/- {final_e_aft_std:.1f}% | "
        f"Net: {final_net_change:+.1f} +/- {final_net_std:.1f}%"
    )

    # Single report at the end because test is evaluated once post-training.
    trial.report(final_success_rate, EPOCHS_PER_TRIAL)

    # Report custom metrics to Optuna so you can view them later
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

    return final_success_rate


def main():
    print("Starting Bayesian Hyperparameter Optimization...")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )

    study.optimize(objective, n_trials=N_TRIALS)

    # Build a complete per-trial view so manual comparison is easy.
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"hyperparam_trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    rows = []
    for tr in study.trials:
        rows.append({
            "trial": tr.number,
            "state": str(tr.state),
            "value": tr.value,
            "lr": tr.params.get("lr"),
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
            "trial", "state", "value", "lr", "massive_reward", "early_finish_bonus",
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
    print("All Trial Results (for manual review):")
    print("Trial | State | SR(mean+/-std) | lr | massive_reward | early_finish_bonus | TotalChange(mean+/-std) | EmptyBefore(mean+/-std) | EmptyAfter(mean+/-std) | Net(mean+/-std)")
    for r in rows:
        sr_std = r["success_rate_std_pct"]
        sr = "-" if r["value"] is None else f"{r['value']:.2f}+/-{(0.0 if sr_std is None else sr_std):.2f}"
        lr = "-" if r["lr"] is None else f"{r['lr']:.2e}"
        mr = "-" if r["massive_reward"] is None else f"{r['massive_reward']:.1f}"
        eb = "-" if r["early_finish_bonus"] is None else f"{r['early_finish_bonus']:.1f}"
        tc = "-" if r["total_change_pct"] is None else f"{r['total_change_pct']:.2f}+/-{(0.0 if r['total_change_std_pct'] is None else r['total_change_std_pct']):.2f}"
        e0 = "-" if r["empty_before_pct"] is None else f"{r['empty_before_pct']:.2f}+/-{(0.0 if r['empty_before_std_pct'] is None else r['empty_before_std_pct']):.2f}"
        e1 = "-" if r["empty_after_pct"] is None else f"{r['empty_after_pct']:.2f}+/-{(0.0 if r['empty_after_std_pct'] is None else r['empty_after_std_pct']):.2f}"
        ne = "-" if r["net_empty_change_pct"] is None else f"{r['net_empty_change_pct']:+.2f}+/-{(0.0 if r['net_empty_change_std_pct'] is None else r['net_empty_change_std_pct']):.2f}"
        print(
            f"{r['trial']:>5} | {r['state']:<7} | {sr:>6} | {lr:>10} | {mr:>14} | {eb:>17} | {tc:>13} | {e0:>14} | {e1:>13} | {ne:>10}"
        )

    print(f"\nSaved full trial table to: {csv_path}")

    if study.best_trial is not None:
        print(f"Best Trial: #{study.best_trial.number}")
        print(
            f"Best Success Rate: {study.best_value:.2f} +/- "
            f"{study.best_trial.user_attrs.get('Success Rate Std %', 0):.2f}%"
        )
        print("Best Parameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")

        print("Metrics for Best Trial:")
        print(
            f"  Total Change:  {study.best_trial.user_attrs.get('Total Change %', 0):.2f} +/- "
            f"{study.best_trial.user_attrs.get('Total Change Std %', 0):.2f}%"
        )
        print(
            f"  Empty Before:  {study.best_trial.user_attrs.get('Empty % Before', 0):.2f} +/- "
            f"{study.best_trial.user_attrs.get('Empty % Before Std', 0):.2f}%"
        )
        print(
            f"  Empty After:   {study.best_trial.user_attrs.get('Empty % After', 0):.2f} +/- "
            f"{study.best_trial.user_attrs.get('Empty % After Std', 0):.2f}%"
        )
        print(
            f"  Net Change:    {study.best_trial.user_attrs.get('Net Empty Change %', 0):+.2f} +/- "
            f"{study.best_trial.user_attrs.get('Net Empty Change Std %', 0):.2f}%"
        )


if __name__ == "__main__":
    main()