"""
finetune_pg.py
==============
Fine-tunes the D3PM Theory Denoiser using Policy Gradient Reinforcement Learning.
Optimized for Minimal-Edit Theory Revision.
"""

import os
from datetime import datetime
from pathlib import Path

# Prevent PyTorch from holding onto fragmented memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.amp import autocast

# Import your core modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from main import (
    TheoryDenoiserNet,
    D3PMForwardCorruption,
    get_dataloaders,
    is_consistent,
    resolve_curriculum_bounds
)

# ==========================================
# Optimal Finetuning Configuration
# ==========================================
CHECKPOINT_PATH = "outputs/theory_denoiser_base.pt"
NUM_EPOCHS = 5000
NUM_ROLLOUTS_PER_EPOCH = 16
SAVE_INTERVAL = 500  # Save a checkpoint every 500 epochs
START_DENOISE_MIN, START_DENOISE_MAX = 50, 251
DATA_REGEN_INTERVAL = 100  # Generate fresh theories every 100 epochs

# Winning Hyperparameters from Optuna Trial 17
PARAMS = {
    'lr': 3.76e-5,                     # Starting Learning Rate
    'change_penalty': 1.0,          # The standard Edit Tax
    'empty_penalty': 4.5,           # The Anti-Erasure Tax
    'massive_reward': 195.0,        # Consistency Payout
    'early_finish_bonus': 47.0,      # Focus purely on spatial edits
}

# ==========================================
# Logging Setup
# ==========================================
log_lines: list[str] = []

def emit(msg: str) -> None:
    print(msg)
    log_lines.append(msg)


def run_rl_epoch(model, corrupt, optimizer, device, params, x_0_batch, clause_mask_batch, update_model=True):
    if update_model:
        model.train()
    else:
        model.eval()

    batch_size = x_0_batch.size(0)
    clause_mask = clause_mask_batch.to(device)
    valid_mask_float = clause_mask.unsqueeze(1).float()

    # SDEdit Partial Corruption [50, 250]
    start_t = torch.randint(START_DENOISE_MIN, START_DENOISE_MAX, (1,)).item()
    t_start_tensor = torch.full((batch_size,), start_t, device=device, dtype=torch.long)

    x_t = corrupt.q_sample(x_0_batch.to(device), t_start_tensor, clause_mask=clause_mask)
    x_t = x_t.masked_fill(~clause_mask.unsqueeze(1), 0)

    x_initial = x_t.clone()
    x_final = torch.zeros_like(x_t)

    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    saved_transitions = []
    traj_rewards = [[] for _ in range(batch_size)]
    hits = 0

    with torch.no_grad():
        for t_step in reversed(range(1, start_t + 1)):
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
                active_theory = x_0_pred[b, :, clause_mask[b]].detach().cpu()

                # --- TERMINAL REWARD LOGIC ---
                if active_theory.numel() > 0 and is_consistent(active_theory):
                    time_saved_pct = (t_step - 1) / start_t
                    gross_reward = params['massive_reward'] + (params['early_finish_bonus'] * time_saved_pct)

                    valid_cells_b = valid_mask_float[b].sum().item() * model.N

                    # 1. Total Edit Penalty
                    total_changes = ((x_initial[b] != x_0_pred[b]) * valid_mask_float[b]).sum().item()
                    change_pct = (total_changes / max(1.0, valid_cells_b)) * 100.0
                    edit_tax = params['change_penalty'] * change_pct

                    # 2. Empty Penalty (Anti-Erasure)
                    empty_before = ((x_initial[b] == 0) * valid_mask_float[b]).sum().item()
                    empty_after = ((x_0_pred[b] == 0) * valid_mask_float[b]).sum().item()
                    net_empty_pct = ((empty_after - empty_before) / max(1.0, valid_cells_b)) * 100.0

                    empty_tax = params['empty_penalty'] * max(0.0, net_empty_pct)

                    # Apply Taxes
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

    # Metric Calculation
    valid_cells = (valid_mask_float.sum(dim=(1, 2)) * model.N).clamp_min(1.0)
    changed_cells = ((x_initial != x_final) * valid_mask_float).sum(dim=(1, 2))
    empty_before_cells = ((x_initial == 0) * valid_mask_float).sum(dim=(1, 2))
    empty_after_cells = ((x_final == 0) * valid_mask_float).sum(dim=(1, 2))

    total_change_pct = (changed_cells / valid_cells).mean().item() * 100.0
    empty_pct_before = (empty_before_cells / valid_cells).mean().item() * 100.0
    empty_pct_after = (empty_after_cells / valid_cells).mean().item() * 100.0

    all_returns = []
    batch_returns_lists = []
    for b in range(batch_size):
        total_episode_return = sum(traj_rewards[b])
        returns = [total_episode_return for _ in traj_rewards[b]]
        batch_returns_lists.append(returns)
        if returns: all_returns.extend(returns)

    if not all_returns:
        return 0.0, 0.0, 0.0, total_change_pct, empty_pct_before, empty_pct_after, batch_size

    baseline = sum(all_returns) / len(all_returns)

    if not update_model:
        return 0.0, (hits / batch_size) * 100, baseline, total_change_pct, empty_pct_before, empty_pct_after, batch_size

    optimizer.zero_grad()
    total_policy_loss_val = 0.0
    b_step_indices = [0 for _ in range(batch_size)]
    use_amp = device.type == "cuda"

    for trans in saved_transitions:
        active_curr = trans['active_mask']
        if not active_curr.any(): continue

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


def main():
    emit(f"Loading Base Checkpoint: {CHECKPOINT_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = TheoryDenoiserNet(N=cfg["N_LITERALS"], M=cfg["M_CLAUSES"], num_classes=cfg["K_STATES"], num_timesteps=cfg["NUM_TIMESTEPS"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    corrupt = D3PMForwardCorruption(num_classes=cfg["K_STATES"], num_timesteps=cfg["NUM_TIMESTEPS"]).to(device)

    # Setup Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Resolve Curriculum
    m_stages = cfg.get("CURRICULUM_M_STAGES", cfg.get("CURRICULUM_STAGES", []))
    _, active_max_clauses = resolve_curriculum_bounds(999, [], cfg["N_LITERALS"])
    active_min_clauses, active_max_clauses = resolve_curriculum_bounds(999, m_stages, cfg["M_CLAUSES"])

    emit(f"Generating Initial Datasets...")
    train_loader, test_loader, _ = get_dataloaders(
        num_samples=NUM_ROLLOUTS_PER_EPOCH * cfg["BATCH_SIZE"],
        N=cfg["N_LITERALS"], max_clauses=active_max_clauses,
        batch_size=cfg["BATCH_SIZE"], train_ratio=0.8,
        active_N=cfg["N_LITERALS"], min_active_N=1, min_clauses=active_min_clauses
    )

    emit("\nStarting RL Fine-Tuning...")
    emit(f"Epochs: {NUM_EPOCHS} | Rollouts/Epoch: {NUM_ROLLOUTS_PER_EPOCH} | Batch Size: {cfg['BATCH_SIZE']}")
    emit(f"Targeting SDEdit Revision Window: t in [{START_DENOISE_MIN}, {START_DENOISE_MAX}]\n")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(1, NUM_EPOCHS + 1):

        # Periodically regenerate the data to prevent overfitting
        if epoch > 1 and (epoch - 1) % DATA_REGEN_INTERVAL == 0:
            emit(f"\n[Data Refresh] Regenerating fresh theories at Epoch {epoch}...")
            train_loader, test_loader, _ = get_dataloaders(
                num_samples=NUM_ROLLOUTS_PER_EPOCH * cfg["BATCH_SIZE"],
                N=cfg["N_LITERALS"], max_clauses=active_max_clauses,
                batch_size=cfg["BATCH_SIZE"], train_ratio=0.8,
                active_N=cfg["N_LITERALS"], min_active_N=1, min_clauses=active_min_clauses
            )

        tr_loss, tr_sr, tr_base, tr_chg, tr_net_e = 0.0, 0.0, 0.0, 0.0, 0.0
        batches = 0

        # Training
        for x_0_batch, clause_mask in train_loader:
            loss, sr, base, chg, e_bef, e_aft, bsz = run_rl_epoch(
                model, corrupt, optimizer, device, PARAMS, x_0_batch, clause_mask, update_model=True
            )
            if bsz > 0:
                tr_loss += loss
                tr_sr += sr
                tr_base += base
                tr_chg += chg
                tr_net_e += (e_aft - e_bef)
                batches += 1

        # Step the learning rate scheduler
        scheduler.step()

        if batches > 0:
            tr_loss /= batches
            tr_sr /= batches
            tr_base /= batches
            tr_chg /= batches
            tr_net_e /= batches

        # Validation (Every 10 Epochs)
        if epoch % 10 == 0:
            te_sr, te_chg, te_net_e = 0.0, 0.0, 0.0
            te_batches = 0
            with torch.no_grad():
                for x_0_batch, clause_mask in test_loader:
                    _, sr, _, chg, e_bef, e_aft, bsz = run_rl_epoch(
                        model, corrupt, None, device, PARAMS, x_0_batch, clause_mask, update_model=False
                    )
                    if bsz > 0:
                        te_sr += sr
                        te_chg += chg
                        te_net_e += (e_aft - e_bef)
                        te_batches += 1

            if te_batches > 0:
                te_sr /= te_batches
                te_chg /= te_batches
                te_net_e /= te_batches

            current_lr = scheduler.get_last_lr()[0]
            emit(f"Epoch {epoch:04d}/{NUM_EPOCHS} | LR: {current_lr:.2e} | Loss: {tr_loss:7.2f} | BaseRet: {tr_base:7.2f}")
            emit(f"  Train -> SR: {tr_sr:6.2f}% | Edit: {tr_chg:6.2f}% | NetEmpty: {tr_net_e:+6.2f}%")
            emit(f"  Test  -> SR: {te_sr:6.2f}% | Edit: {te_chg:6.2f}% | NetEmpty: {te_net_e:+6.2f}%")
            emit("-" * 65)

        # Periodic Checkpointing
        if epoch % SAVE_INTERVAL == 0:
            save_path = out_dir / f"theory_denoiser_rl_{run_id}_ep{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": cfg,
                "rl_params": PARAMS
            }, save_path)
            emit(f">>> Checkpoint saved: {save_path.name}")

    # Final Checkpoint and Log Persist
    final_path = out_dir / f"theory_denoiser_rl_{run_id}_FINAL.pt"
    log_path = out_dir / f"finetune_log_{run_id}.txt"

    torch.save({
        "epoch": NUM_EPOCHS,
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "rl_params": PARAMS
    }, final_path)

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    emit(f"\nFinetuning Complete! Final model saved to {final_path}")
    emit(f"Saved training log: {log_path}")

if __name__ == "__main__":
    main()