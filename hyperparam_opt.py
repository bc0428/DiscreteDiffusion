"""
tune_rl_hyperparams.py
======================
Optimizes RL reward parameters and learning rate for the D3PM Theory Denoiser
using Bayesian Optimization (Optuna).

Target Metric: Maximize the Consistency Success Rate.
"""

import os

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
EPOCHS_PER_TRIAL = 5  # Keep this low to iterate quickly
NUM_ROLLOUTS_PER_EPOCH = 16  # Reduced for faster trial evaluation
N_TRIALS = 50  # Total number of hyperparameter combinations to test


def run_tuning_rollouts(model, corrupt, dataloader, optimizer, device, params):
    """
    Your optimized rollout logic, adapted to accept dynamic parameters from Optuna.
    """
    model.train()
    try:
        _, clause_mask = next(iter(dataloader))
    except StopIteration:
        return 0, 0, 0

    batch_size = clause_mask.size(0)
    clause_mask = clause_mask.to(device)
    valid_mask_float = clause_mask.unsqueeze(1).float()

    x_t = torch.randint(0, model.num_classes, (batch_size, model.N, clause_mask.size(1)), device=device)
    x_t = x_t.masked_fill(~clause_mask.unsqueeze(1), 0)

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

    # ── PHASE 2: CALCULATE RETURNS ──
    all_returns = []
    batch_returns_lists = []

    for b in range(batch_size):
        # G = 0.0
        returns = []
        # for r in reversed(traj_rewards[b]):
        #     G = r + params['gamma'] * G
        #     returns.insert(0, G)
        returns.insert(0, sum(traj_rewards[b]))  # Total return at the start of the episode
        batch_returns_lists.append(returns)
        if returns:
            all_returns.extend(returns)

    if not all_returns:
        return 0.0, 0, 0.0

    baseline = sum(all_returns) / len(all_returns)

    # ── PHASE 3: STEP-WISE BACKPROPAGATION ──
    optimizer.zero_grad()
    total_policy_loss_val = 0.0
    b_step_indices = [0 for _ in range(batch_size)]

    for trans in saved_transitions:
        active_curr = trans['active_mask']
        if not active_curr.any():
            continue

        with autocast(device_type='cuda', dtype=torch.bfloat16):
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

    return total_policy_loss_val, (hits / batch_size) * 100, baseline


def objective(trial):
    """
    The Optuna objective function. Suggests parameters, runs a short training loop,
    and returns the final success rate.
    """
    # 1. Define the Search Space
    params = {
        'lr': trial.suggest_float('lr', 1e-6, 5e-5, log=True),
        'step_cost': 1.0,
        'massive_reward': trial.suggest_float('massive_reward', 50.0, 300.0, step=50.0),
        'early_finish_bonus': trial.suggest_float('early_finish_bonus', 1.0, 10.0, step=3.0),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    # 2. Initialize a Fresh Model for this Trial
    model = TheoryDenoiserNet(N=cfg["N_LITERALS"], M=cfg["M_CLAUSES"], num_classes=cfg["K_STATES"],
                              num_timesteps=cfg["NUM_TIMESTEPS"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    corrupt = D3PMForwardCorruption(num_classes=cfg["K_STATES"], num_timesteps=cfg["NUM_TIMESTEPS"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    final_max_clauses = resolve_curriculum_max_clauses(999, cfg["CURRICULUM_STAGES"], cfg["M_CLAUSES"])
    train_loader, _, _ = get_dataloaders(
        num_samples=NUM_ROLLOUTS_PER_EPOCH * cfg["BATCH_SIZE"],
        N=cfg["N_LITERALS"], max_clauses=final_max_clauses,
        batch_size=cfg["BATCH_SIZE"], train_ratio=1.0
    )

    # 3. Mini Training Loop
    final_success_rate = 0.0

    for epoch in range(1, EPOCHS_PER_TRIAL + 1):
        epoch_sr = 0.0
        for _ in range(NUM_ROLLOUTS_PER_EPOCH):
            _, sr, _ = run_tuning_rollouts(model, corrupt, train_loader, optimizer, device, params)
            epoch_sr += sr

        avg_epoch_sr = epoch_sr / NUM_ROLLOUTS_PER_EPOCH
        final_success_rate = avg_epoch_sr

        # Report intermediate values to Optuna for pruning
        trial.report(avg_epoch_sr, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Free up VRAM before the next trial starts
    del model
    del optimizer
    torch.cuda.empty_cache()

    return final_success_rate


def main():
    print("Starting Bayesian Hyperparameter Optimization...")

    # Create an Optuna study that maximizes the success rate
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "=" * 50)
    print("OPTIMIZATION FINISHED")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Success Rate: {study.best_value:.2f}%")
    print("Best Parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()