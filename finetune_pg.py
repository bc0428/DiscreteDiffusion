"""
finetune_pg.py
==============
Fine-tunes a trained TheoryDenoiserNet using standard Policy Gradient (REINFORCE).
Simulates full reverse-diffusion rollouts and optimizes based on cumulative discounted returns.

Rewards:
1. Step Cost: Penalty for changes between x_t and x_{t-1}.
2. Consistency Reward: Massive payout if is_consistent(x_0_pred) triggers.
3. Early Finish Bonus: Multiplier * remaining steps if consistency is hit early.
"""

import os
# Must be set before importing PyTorch to prevent memory fragmentation on 8GB GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.amp import autocast

# ── Import training artifacts from main.py ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from main import (
    TheoryDenoiserNet,
    D3PMForwardCorruption,
    get_dataloaders,
    is_consistent,
    resolve_curriculum_max_clauses
)

# ==========================================
# Configuration Block
# ==========================================
CHECKPOINT_PATH = "outputs/theory_denoiser_base.pt"
EPOCHS = 20
LR = 5e-6
NUM_ROLLOUTS_PER_EPOCH = 32

# RL Parameters
GAMMA = 0.99
STEP_COST = 0.5
MASSIVE_REWARD = 100.0
EARLY_FINISH_BONUS = 5.0


def run_pg_rollouts(model, corrupt, dataloader, optimizer, device):
    model.train()

    try:
        _, clause_mask = next(iter(dataloader))
    except StopIteration:
        return 0, 0, 0

    batch_size = clause_mask.size(0)
    clause_mask = clause_mask.to(device)
    valid_mask_float = clause_mask.unsqueeze(1).float()

    # Start from pure noise
    x_t = torch.randint(0, model.num_classes, (batch_size, model.N, clause_mask.size(1)), device=device)
    x_t = x_t.masked_fill(~clause_mask.unsqueeze(1), 0)

    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    # Track trajectories for decoupled gradient computation
    saved_transitions = []
    traj_rewards = [[] for _ in range(batch_size)]
    hits = 0

    # =========================================================================
    # PHASE 1: NO-GRADIENT ROLLOUT (Memory Saver)
    # =========================================================================
    with torch.no_grad():
        for t_step in reversed(range(1, model.num_timesteps + 1)):
            if not active_mask.any():
                break

            t_tensor = torch.full((batch_size,), t_step, device=device, dtype=torch.long)

            # Save the state exactly as it was at the start of this step
            current_transition = {
                'x_t': x_t.clone(),
                't_tensor': t_tensor.clone(),
                'active_mask': active_mask.clone()
            }

            logits = model(x_t, t_tensor, clause_mask=clause_mask)
            dist = torch.distributions.Categorical(logits=logits)
            x_0_pred = dist.sample()
            x_0_pred = x_0_pred.masked_fill(~clause_mask.unsqueeze(1), 0)

            current_transition['x_0_pred'] = x_0_pred.clone()
            saved_transitions.append(current_transition)

            # Generate next state x_{t-1}
            if t_step > 1:
                t_minus_1 = torch.full((batch_size,), t_step - 1, device=device, dtype=torch.long)
                x_t_minus_1 = corrupt.q_sample(x_0_pred, t_minus_1, clause_mask=clause_mask)
            else:
                x_t_minus_1 = x_0_pred

            # Evaluate Step Rewards

            for b in range(batch_size):
                if not active_mask[b]:
                    continue

                r=0
                active_theory = x_0_pred[b, :, clause_mask[b]].detach().cpu()

                if active_theory.numel() > 0 and is_consistent(active_theory):
                    r = MASSIVE_REWARD + (EARLY_FINISH_BONUS * (t_step - 1))
                    active_mask[b] = False
                    hits += 1

                changes = ((x_t[b] != x_t_minus_1[b]) * valid_mask_float[b]).sum().item()
                r += - (STEP_COST * changes)
                traj_rewards[b].append(r)

            x_t = x_t_minus_1

    # =========================================================================
    # PHASE 2: CALCULATE RETURNS & BASELINE
    # =========================================================================
    all_returns = []
    batch_returns_lists = []

    for b in range(batch_size):
        G = 0.0
        returns = []
        for r in reversed(traj_rewards[b]):
            G = r + GAMMA * G
            returns.insert(0, G)
        batch_returns_lists.append(returns)
        if returns:
            all_returns.extend(returns)

    if not all_returns:
        return 0.0, 0, 0.0

    baseline = sum(all_returns) / len(all_returns)

    # =========================================================================
    # PHASE 3: STEP-WISE BACKPROPAGATION (VRAM Optimization)
    # =========================================================================
    optimizer.zero_grad()
    total_policy_loss_val = 0.0

    # Track which step each batch item is on to map to the correct return
    b_step_indices = [0 for _ in range(batch_size)]

    for step_idx, trans in enumerate(saved_transitions):
        active_curr = trans['active_mask']
        if not active_curr.any():
            continue

        # Use Mixed Precision for massive memory and speed gains on Blackwell
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(trans['x_t'], trans['t_tensor'], clause_mask=clause_mask)
            dist = torch.distributions.Categorical(logits=logits)

            step_log_probs = (dist.log_prob(trans['x_0_pred']) * valid_mask_float).sum(dim=(1,2))

            step_loss = 0.0
            has_loss = False

            for b in range(batch_size):
                if active_curr[b]:
                    G_t = batch_returns_lists[b][b_step_indices[b]]
                    b_step_indices[b] += 1

                    advantage = G_t - baseline
                    # REINFORCE objective: maximize expected return
                    step_loss = step_loss - (step_log_probs[b] * advantage) / batch_size
                    has_loss = True

        # BACKPROPAGATE IMMEDIATELY: This frees the Transformer graph from VRAM
        # before the next diffusion step is calculated.
        if has_loss:
            step_loss.backward()
            total_policy_loss_val += step_loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    avg_return = baseline
    success_rate = (hits / batch_size) * 100

    return total_policy_loss_val, success_rate, avg_return


def main():
    # ── 1. Load Checkpoint ───────────────────────────────────────────────────
    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        print(f"ERROR: Could not find checkpoint at {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 2. Initialize Models ─────────────────────────────────────────────────
    model = TheoryDenoiserNet(N=cfg["N_LITERALS"], M=cfg["M_CLAUSES"], num_classes=cfg["K_STATES"], num_timesteps=cfg["NUM_TIMESTEPS"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    corrupt = D3PMForwardCorruption(num_classes=cfg["K_STATES"], num_timesteps=cfg["NUM_TIMESTEPS"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── 3. Data Loaders (For structural masks only) ──────────────────────────
    final_max_clauses = resolve_curriculum_max_clauses(999, cfg["CURRICULUM_STAGES"], cfg["M_CLAUSES"])
    train_loader, _, _ = get_dataloaders(
        num_samples=NUM_ROLLOUTS_PER_EPOCH * cfg["BATCH_SIZE"],
        N=cfg["N_LITERALS"], max_clauses=final_max_clauses,
        batch_size=cfg["BATCH_SIZE"], train_ratio=1.0
    )

    # ── 4. Training Loop ──────────────────────────────────────────────────────
    print(f"\nStarting REINFORCE Fine-tuning on device: {device}")
    print(f"Params: Gamma={GAMMA}, LR={LR}, StepCost={STEP_COST}, ConsistReward={MASSIVE_REWARD}, EarlyBonus={EARLY_FINISH_BONUS}\n")

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        epoch_sr = 0.0
        epoch_return = 0.0

        for _ in range(NUM_ROLLOUTS_PER_EPOCH):
            loss, sr, avg_ret = run_pg_rollouts(model, corrupt, train_loader, optimizer, device)
            epoch_loss += loss
            epoch_sr += sr
            epoch_return += avg_ret

        print(f"Epoch {epoch:>2}/{EPOCHS} | Loss: {epoch_loss/NUM_ROLLOUTS_PER_EPOCH:.4f} | "
              f"Success Rate: {epoch_sr/NUM_ROLLOUTS_PER_EPOCH:.1f}% | Avg Return: {epoch_return/NUM_ROLLOUTS_PER_EPOCH:.2f}")

    # ── 5. Save Model ─────────────────────────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"theory_denoiser_pg_finetuned_{run_id}.pt"

    cfg["PG_FINETUNED"] = True
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": cfg}, save_path)
    print(f"\nSaved PG fine-tuned model checkpoint to: {save_path}")

if __name__ == "__main__":
    main()