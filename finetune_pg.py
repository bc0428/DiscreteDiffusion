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
from torch.utils.data import Dataset, DataLoader

# Import your core modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from main import (
    TheoryDenoiserNet,
    D3PMForwardCorruption,
    is_consistent,
)

# ==========================================
# Optimal Finetuning Configuration
# ==========================================
CHECKPOINT_PATH = "outputs/theory_denoiser_base.pt"
NUM_EPOCHS = 5000
NUM_ROLLOUTS_PER_EPOCH = 16
SAVE_INTERVAL = 500  # Save a checkpoint every 500 epochs
VALIDATION_INTERVAL = 25000  # Run validation every N epochs
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

# Winning Hyperparameters from Optuna Trial 17
PARAMS = {
    'lr': 3e-5,                     # Starting Learning Rate
    'change_penalty': 0.17337,      # The standard Edit Tax
    'empty_penalty': 12.0,          # The Anti-Erasure Tax
    'massive_reward': 260.0,        # Consistency Payout
    'early_finish_bonus': 6.0,      # Focus purely on spatial edits
}

# ==========================================
# Logging Setup (MOVED TO TOP FOR SCOPE)
# ==========================================
log_lines: list[str] = []

def emit(msg: str) -> None:
    print(msg)
    log_lines.append(msg)


# ==========================================
# Data Handling & Globals
# ==========================================
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


# GLOBAL CACHE TO PREVENT RAM EXPLOSION
TRAIN_DATA_CACHE = None
VAL_DATA_CACHE = None

def load_hyperparam_dataloaders(batch_size: int, num_timesteps: int, start_pct_min: float = None, start_pct_max: float = None):
    global TRAIN_DATA_CACHE, VAL_DATA_CACHE

    if start_pct_min is None:
        start_pct_min = START_DENOISE_TRAJ_PCT_MIN
    if start_pct_max is None:
        start_pct_max = START_DENOISE_TRAJ_PCT_MAX

    # ONLY load from disk if it hasn't been loaded yet
    if TRAIN_DATA_CACHE is None:
        emit("Loading massive hyperparam train dataset into RAM (This happens ONLY ONCE)...")
        TRAIN_DATA_CACHE = torch.load(DATASET_DIR / "hyperparam_train.pt", weights_only=False)
    if VAL_DATA_CACHE is None:
        emit("Loading massive hyperparam val dataset into RAM (This happens ONLY ONCE)...")
        VAL_DATA_CACHE = torch.load(DATASET_DIR / "hyperparam_val.pt", weights_only=False)

    train_dataset = HyperparamDataset(TRAIN_DATA_CACHE, num_timesteps=num_timesteps, start_traj_pct_min=start_pct_min, start_traj_pct_max=start_pct_max)
    val_dataset = HyperparamDataset(VAL_DATA_CACHE, num_timesteps=num_timesteps, start_traj_pct_min=start_pct_min, start_traj_pct_max=start_pct_max)

    # SHUFFLE IS FALSE: Load sequentially since the dataset is already randomized in generation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=theory_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=theory_collate_fn, num_workers=0)

    return train_loader, val_loader


def get_infinite_batches(dataloader):
    """Yields batches sequentially and loops back to the start automatically."""
    while True:
        for batch in dataloader:
            yield batch


# ==========================================
# Environment Step Logic
# ==========================================
def run_rl_epoch(model, corrupt, optimizer, device, params, x_0_batch, x_start_batch, clause_mask_batch, start_t_batch, update_model=True):
    if update_model:
        model.train()
    else:
        model.eval()

    batch_size = x_0_batch.size(0)
    clause_mask = clause_mask_batch.to(device)
    valid_mask_float = clause_mask.unsqueeze(1).float()
    start_t_per_sample = start_t_batch.to(device).long()
    max_start_t = int(start_t_per_sample.max().item())

    x_t = x_start_batch.to(device)
    x_t = x_t.masked_fill(~clause_mask.unsqueeze(1), 0)

    x_initial = x_t.clone()
    x_final = torch.zeros_like(x_t)

    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    saved_transitions = []
    traj_rewards = [[] for _ in range(batch_size)]
    hits = 0

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
                active_theory = x_0_pred[b, :, clause_mask[b]].detach().cpu()

                # --- TERMINAL REWARD LOGIC ---
                if active_theory.numel() > 0 and is_consistent(active_theory):
                    start_t_b = max(1, int(start_t_per_sample[b].item()))
                    time_saved_pct = (t_step - 1) / start_t_b
                    gross_reward = params['massive_reward'] + (params['early_finish_bonus'] * time_saved_pct)

                    valid_cells_b = valid_mask_float[b].sum().item() * model.N
                    total_changes = ((x_initial[b] != x_0_pred[b]) * valid_mask_float[b]).sum().item()

                    # 1. Edit Tax
                    change_ratio = total_changes / max(1.0, valid_cells_b)
                    edit_tax = params['change_penalty'] * (change_ratio ** 1.5) * 100.0

                    # 2. Empty Penalty
                    empty_before = ((x_initial[b] == 0) * valid_mask_float[b]).sum().item()
                    empty_after = ((x_0_pred[b] == 0) * valid_mask_float[b]).sum().item()

                    net_empty_ratio = (empty_after - empty_before) / max(1.0, valid_cells_b)
                    net_empty_ratio = max(0.0, net_empty_ratio)
                    empty_tax = params['empty_penalty'] * net_empty_ratio * 100.0

                    r = gross_reward - edit_tax - empty_tax
                    active_mask[b] = False
                    hits += 1
                else:
                    # 3. Critical Failure Penalty
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
        if returns:
            all_returns.extend(returns)

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


# ==========================================
# Main Finetuning Loop
# ==========================================
def main():
    ckpt_path = resolve_base_checkpoint()
    emit(f"Loading Base Checkpoint: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = TheoryDenoiserNet(N=cfg["N_LITERALS"], M=cfg["M_CLAUSES"], num_classes=cfg["K_STATES"], num_timesteps=cfg["NUM_TIMESTEPS"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    corrupt = D3PMForwardCorruption(num_classes=cfg["K_STATES"], num_timesteps=cfg["NUM_TIMESTEPS"]).to(device)

    # Setup Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Load fixed finetune datasets
    emit(f"Loading Finetune Datasets from disk...")
    train_loader, test_loader = load_hyperparam_dataloaders(batch_size=cfg["BATCH_SIZE"], num_timesteps=cfg["NUM_TIMESTEPS"])

    # INFINITE ITERATOR SETUP
    batches_per_epoch = max(1, NUM_ROLLOUTS_PER_EPOCH // cfg["BATCH_SIZE"])
    train_iter = iter(get_infinite_batches(train_loader))

    emit("\nStarting RL Fine-Tuning...")
    emit(f"Epochs: {NUM_EPOCHS} | Rollouts/Epoch: {NUM_ROLLOUTS_PER_EPOCH} | Batch Size: {cfg['BATCH_SIZE']} | Batches/Epoch: {batches_per_epoch}")
    start_t_min = max(1, int(round(START_DENOISE_TRAJ_PCT_MIN * (cfg["NUM_TIMESTEPS"] - 1))))
    start_t_max = max(1, min(cfg["NUM_TIMESTEPS"] - 1, int(round(START_DENOISE_TRAJ_PCT_MAX * (cfg["NUM_TIMESTEPS"] - 1)))))
    emit(f"Trajectory start range: {START_DENOISE_TRAJ_PCT_MIN:.2f}-{START_DENOISE_TRAJ_PCT_MAX:.2f} (approx t={start_t_min}-{start_t_max}/{cfg['NUM_TIMESTEPS'] - 1})\n")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(1, NUM_EPOCHS + 1):

        tr_loss, tr_sr, tr_base, tr_chg, tr_net_e = 0.0, 0.0, 0.0, 0.0, 0.0
        batches = 0

        # --- SEQUENTIAL TRAINING BATCHING ---
        for _ in range(batches_per_epoch):
            x_0_batch, x_start_batch, clause_mask, start_t_batch = next(train_iter)

            loss, sr, base, chg, e_bef, e_aft, bsz = run_rl_epoch(
                model, corrupt, optimizer, device, PARAMS, x_0_batch, x_start_batch, clause_mask, start_t_batch, update_model=True
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

        # Validation (periodic)
        if epoch % VALIDATION_INTERVAL == 0:
            te_sr, te_chg, te_net_e = 0.0, 0.0, 0.0
            te_batches = 0
            with torch.no_grad():
                for x_0_batch, x_start_batch, clause_mask, start_t_batch in test_loader:
                    _, sr, _, chg, e_bef, e_aft, bsz = run_rl_epoch(
                        model, corrupt, None, device, PARAMS, x_0_batch, x_start_batch, clause_mask, start_t_batch, update_model=False
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