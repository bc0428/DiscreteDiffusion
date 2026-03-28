"""
finetune_pg.py
==============
Fine-tunes a pre-trained D3PM Theory Denoiser using Policy Gradient RL.
Uses the same rollout and reward logic as hyperparam_opt.py.
"""

import os
from datetime import datetime
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.amp import autocast

from main import (
    TheoryDenoiserNet,
    D3PMForwardCorruption,
    get_dataloaders,
    is_consistent,
    resolve_curriculum_bounds
)

# ==========================================
# RL Fine-tuning Configuration
# ==========================================
CHECKPOINT_PATH = "outputs/theory_denoiser_base.pt"
EPOCHS = 40
ROLLOUTS_PER_EPOCH = 16

# Update these with the best parameters from your Optuna study
LR = 2.95e-5
CHANGE_PENALTY = 1.91
MASSIVE_REWARD = 116
EARLY_FINISH_BONUS = 44


def run_rollout(model, corrupt, dataloader, optimizer, device, params, update_model=True, clause_mask_batch=None):
    """
    Identical to run_tuning_rollouts in hyperparam_opt.py.
    Runs one rollout (one batch) with the given hyperparameters.
    Returns: (loss, sr, baseline, total_change_pct, empty_pct_before, empty_pct_after, batch_size)
    """
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

                x_final[b] = x_0_pred[b].clone()

                r = 0.0
                active_theory = x_0_pred[b, :, clause_mask[b]].detach().cpu()

                if active_theory.numel() > 0 and is_consistent(active_theory):
                    time_saved_pct = (t_step - 1) / model.num_timesteps
                    gross_reward = params['massive_reward'] + (params['early_finish_bonus'] * time_saved_pct)

                    valid_cells_b = valid_mask_float[b].sum().item() * model.N
                    total_changes = ((x_initial[b] != x_0_pred[b]) * valid_mask_float[b]).sum().item()
                    change_pct = (total_changes / max(1.0, valid_cells_b)) * 100.0

                    edit_tax = params['change_penalty'] * change_pct

                    r = gross_reward - edit_tax
                    active_mask[b] = False
                    hits += 1

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


import math

def mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)

def main():
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = TheoryDenoiserNet(N=cfg["N_LITERALS"], M=cfg["M_CLAUSES"], num_classes=cfg["K_STATES"],
                              num_timesteps=cfg["NUM_TIMESTEPS"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    corrupt = D3PMForwardCorruption(num_classes=cfg["K_STATES"], num_timesteps=cfg["NUM_TIMESTEPS"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    params = {
        'lr': LR,
        'change_penalty': CHANGE_PENALTY,
        'massive_reward': MASSIVE_REWARD,
        'early_finish_bonus': EARLY_FINISH_BONUS,
    }

    # ── NEW CURRICULUM BOUNDS LOGIC ──
    n_stages = cfg.get("CURRICULUM_N_STAGES", [(1, cfg["N_LITERALS"])])
    m_stages = cfg.get("CURRICULUM_M_STAGES", cfg.get("CURRICULUM_STAGES", [(1, cfg["M_CLAUSES"])]))

    active_min_literals, active_max_literals = resolve_curriculum_bounds(999, n_stages, cfg["N_LITERALS"])
    active_min_clauses, active_max_clauses = resolve_curriculum_bounds(999, m_stages, cfg["M_CLAUSES"])

    print(f"Generating {ROLLOUTS_PER_EPOCH * cfg['BATCH_SIZE']} theories for RL environment (Max M={active_max_clauses})...")

    train_loader, val_loader, _ = get_dataloaders(
        num_samples=ROLLOUTS_PER_EPOCH * cfg["BATCH_SIZE"],
        N=cfg["N_LITERALS"],
        max_clauses=active_max_clauses,
        batch_size=cfg["BATCH_SIZE"],
        train_ratio=0.8,
        active_N=active_max_literals,
        min_active_N=active_min_literals,
        min_clauses=active_min_clauses
    )

    log_lines = []

    def emit(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    emit(f"\n--- Starting RL Fine-tuning ---")
    emit(f"RL Hyperparams: LR={LR}, CHANGE_PENALTY={CHANGE_PENALTY}, MASSIVE_REWARD={MASSIVE_REWARD}, EARLY_FINISH_BONUS={EARLY_FINISH_BONUS}")

    # ... [Keep the rest of the train/val loop exactly as previously generated!] ...

if __name__ == "__main__":
    main()