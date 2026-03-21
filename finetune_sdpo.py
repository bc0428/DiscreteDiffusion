"""
finetune_sdpo.py
================
Fine-tunes a trained TheoryDenoiserNet using Step-aware Diffusion Preference
Optimization (SDPO) directly from an IDE.

Rewards are defined as:
  R(x_0) = lambda_consist * (1 if is_consistent(x_0) else 0)
         - lambda_diff * (percentage of cells changed relative to original)
"""

import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F

# ── Import training artifacts ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from main import (
    TheoryDenoiserNet,
    D3PMForwardCorruption,
    get_dataloaders,
    is_consistent,
    resolve_curriculum_max_clauses
)

# ==========================================
# IDE Configuration Block
# ==========================================
CHECKPOINT_PATH = "outputs/theory_denoiser_20260319_122658.pt"  # <-- UPDATE THIS
EPOCHS = 10
LR = 1e-5
BETA = 0.1
LAMBDA_CONSIST = 1.0
LAMBDA_DIFF = 0.5


def compute_reward(x_0_pred: torch.Tensor, x_0_orig: torch.Tensor, clause_mask: torch.Tensor,
                   lambda_consist: float, lambda_diff: float) -> torch.Tensor:
    """
    Computes the scalar reward for a batch of predicted theories.
    """
    batch_size = x_0_pred.size(0)
    rewards = torch.zeros(batch_size, device=x_0_pred.device)

    for b in range(batch_size):
        # Extract active subset using the clause mask
        active_pred = x_0_pred[b, :, clause_mask[b]].cpu()
        active_orig = x_0_orig[b, :, clause_mask[b]].cpu()

        # 1. Consistency Reward
        consist_reward = lambda_consist if (active_pred.numel() > 0 and is_consistent(active_pred)) else 0.0

        # 2. Minimal Change Penalty (Cost for total change)
        total_cells = float(max(1, active_orig.numel()))
        changed_cells = float((active_pred != active_orig).sum().item())
        diff_penalty = lambda_diff * (changed_cells / total_cells)

        rewards[b] = consist_reward - diff_penalty

    return rewards

def get_log_probs(logits: torch.Tensor, target_x_0: torch.Tensor, clause_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the sum of log probabilities for a specific x_0 completion,
    ignoring padded clauses.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = torch.gather(log_probs, dim=-1, index=target_x_0.unsqueeze(-1)).squeeze(-1)

    valid_mask = clause_mask.unsqueeze(1).float()
    return (selected_log_probs * valid_mask).sum(dim=(1, 2))

def run_sdpo_epoch(policy_model, ref_model, corrupt, dataloader, optimizer, beta, lambda_c, lambda_d):
    policy_model.train()
    ref_model.eval()

    total_loss = 0.0
    valid_pairs_count = 0

    for x_0_orig, clause_mask in dataloader:
        batch_size = x_0_orig.size(0)
        device = next(policy_model.parameters()).device

        x_0_orig = x_0_orig.to(device)
        clause_mask = clause_mask.to(device)

        # 1. Sample timestep and add noise
        t = torch.randint(1, corrupt.num_timesteps, (batch_size,), device=device).long()
        x_t = corrupt.q_sample(x_0_orig, t, clause_mask=clause_mask)

        # 2. Generate two completions using the reference model
        with torch.no_grad():
            ref_logits = ref_model(x_t, t, clause_mask)
            dist = torch.distributions.Categorical(logits=ref_logits)

            x_0_1 = dist.sample()
            x_0_2 = dist.sample()

            x_0_1 = x_0_1.masked_fill(~clause_mask.unsqueeze(1), 0)
            x_0_2 = x_0_2.masked_fill(~clause_mask.unsqueeze(1), 0)

        # 3. Calculate Rewards
        r1 = compute_reward(x_0_1, x_0_orig, clause_mask, lambda_c, lambda_d).to(device)
        r2 = compute_reward(x_0_2, x_0_orig, clause_mask, lambda_c, lambda_d).to(device)

        # 4. Identify Winners and Losers
        valid_mask = (r1 != r2)
        if not valid_mask.any():
            continue

        w_mask = (r1 > r2).unsqueeze(-1).unsqueeze(-1)
        x_w = torch.where(w_mask, x_0_1, x_0_2)
        x_l = torch.where(w_mask, x_0_2, x_0_1)

        # 5. Compute Log Probs
        optimizer.zero_grad()
        policy_logits = policy_model(x_t, t, clause_mask)

        with torch.no_grad():
            ref_log_prob_w = get_log_probs(ref_logits, x_w, clause_mask)
            ref_log_prob_l = get_log_probs(ref_logits, x_l, clause_mask)

        policy_log_prob_w = get_log_probs(policy_logits, x_w, clause_mask)
        policy_log_prob_l = get_log_probs(policy_logits, x_l, clause_mask)

        # 6. SDPO Loss Calculation
        pi_logratios = policy_log_prob_w - policy_log_prob_l
        ref_logratios = ref_log_prob_w - ref_log_prob_l

        logits = pi_logratios - ref_logratios
        loss_per_item = -F.logsigmoid(beta * logits)

        loss = loss_per_item[valid_mask].mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        valid_pairs_count += valid_mask.sum().item()

    avg_loss = total_loss / len(dataloader) if valid_pairs_count > 0 else 0.0
    return avg_loss, valid_pairs_count

def main():
    # ── 1. Load Checkpoint & Configuration ────────────────────────────────────
    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        print(f"ERROR: Could not find checkpoint at {ckpt_path}")
        print("Please update CHECKPOINT_PATH in the configuration block.")
        sys.exit(1)

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 2. Initialize Reference and Policy Models ─────────────────────────────
    ref_model = TheoryDenoiserNet(N=cfg["N_LITERALS"], M=cfg["M_CLAUSES"], num_classes=cfg["K_STATES"], num_timesteps=cfg["NUM_TIMESTEPS"])
    ref_model.load_state_dict(ckpt["model_state_dict"])
    ref_model.to(device)
    ref_model.eval()

    policy_model = TheoryDenoiserNet(N=cfg["N_LITERALS"], M=cfg["M_CLAUSES"], num_classes=cfg["K_STATES"], num_timesteps=cfg["NUM_TIMESTEPS"])
    policy_model.load_state_dict(ckpt["model_state_dict"])
    policy_model.to(device)

    corrupt = D3PMForwardCorruption(num_classes=cfg["K_STATES"], num_timesteps=cfg["NUM_TIMESTEPS"]).to(device)

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=LR)

    # ── 3. Data Preparation ───────────────────────────────────────────────────
    print("Generating data for fine-tuning...")
    final_max_clauses = resolve_curriculum_max_clauses(999, cfg["CURRICULUM_STAGES"], cfg["M_CLAUSES"])

    train_loader, _, _ = get_dataloaders(
        num_samples=cfg["NUM_SAMPLES"],
        N=cfg["N_LITERALS"],
        max_clauses=final_max_clauses,
        batch_size=cfg["BATCH_SIZE"],
        train_ratio=1.0
    )

    # ── 4. Training Loop ──────────────────────────────────────────────────────
    print(f"\nStarting SDPO Fine-tuning on device: {device}")
    print(f"Params: Beta={BETA}, LR={LR}, L_consist={LAMBDA_CONSIST}, L_diff={LAMBDA_DIFF}")

    for epoch in range(1, EPOCHS + 1):
        loss, valid_pairs = run_sdpo_epoch(
            policy_model=policy_model,
            ref_model=ref_model,
            corrupt=corrupt,
            dataloader=train_loader,
            optimizer=optimizer,
            beta=BETA,
            lambda_c=LAMBDA_CONSIST,
            lambda_d=LAMBDA_DIFF
        )
        print(f"Epoch {epoch:>2}/{EPOCHS} | SDPO Loss: {loss:.4f} | Valid Preference Pairs: {valid_pairs}")

    # ── 5. Save Fine-tuned Model ──────────────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"theory_denoiser_sdpo_finetuned_{run_id}.pt"

    cfg["SDPO_FINETUNED"] = True
    cfg["SDPO_BETA"] = BETA

    torch.save({
        "model_state_dict": policy_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg
    }, save_path)

    print(f"\nSaved fine-tuned model checkpoint to: {save_path}")

if __name__ == "__main__":
    main()