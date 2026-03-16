import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from pysat.solvers import Solver
from torch.utils.data import Dataset, DataLoader, random_split

# SAT backend used by is_consistent; m22 is usually more stable on Windows than g3.
SAT_SOLVER_BACKEND = "m22"


# ==========================================
# Theory Generation & Consistency
# ==========================================

def theory_to_cnf_clauses(theory: torch.Tensor) -> list[list[int]]:
    """
    Convert one theory matrix of shape (N, M) into SAT clauses.

    Convention:
      - each column j is one CNF clause
      - row i corresponds to boolean variable (i + 1)
      - value 1 -> positive literal  +(i + 1)
      - value 2 -> negative literal  -(i + 1)
      - value 0 -> literal absent from that clause

    Returns:
        A list of clauses in PySAT format, e.g. [[1, -2], [3]].
    """
    if theory.dim() != 2:
        raise ValueError(f"Expected a 2D theory matrix of shape (N, M), got shape {tuple(theory.shape)}")

    num_literals, num_clauses = theory.shape
    clauses: list[list[int]] = []

    for j in range(num_clauses):
        clause: list[int] = []
        seen_pos = set()
        seen_neg = set()

        for i in range(num_literals):
            value = int(theory[i, j].item())
            var_id = i + 1

            if value == 1:
                seen_pos.add(var_id)
                clause.append(var_id)
            elif value == 2:
                seen_neg.add(var_id)
                clause.append(-var_id)
            elif value != 0:
                raise ValueError(f"Theory entries must be in {{0,1,2}}, got {value} at ({i}, {j})")

        # Defensive handling: a clause containing both x and ¬x is a tautology,
        # so it can be skipped without affecting satisfiability.
        if seen_pos & seen_neg:
            continue

        clauses.append(clause)

    return clauses


def is_consistent(theory: torch.Tensor) -> bool:
    """
    Check whether the CNF theory encoded by the (N x M) matrix is satisfiable.
    """
    clauses = theory_to_cnf_clauses(theory)

    with Solver(name=SAT_SOLVER_BACKEND) as solver:
        solver.append_formula(clauses)
        return solver.solve()


def generate_consistent_theory(N: int, max_clauses: int) -> torch.Tensor:
    """
    Generate one consistent theory with variable clause length.
    """
    if max_clauses <= 0:
        raise ValueError("max_clauses must be >= 1")

    while True:
        theory = torch.zeros(N, max_clauses, dtype=torch.long)

        for i in range(N):
            polarity = torch.randint(0, 3, (1,)).item()  # 0=absent, 1=pos-only, 2=neg-only
            if polarity == 0:
                continue
            if polarity == 1:
                theory[i] = torch.randint(0, 2, (max_clauses,))  # {0,1}
            else:
                theory[i] = torch.randint(0, 2, (max_clauses,)) * 2  # {0,2}

        # Step 4: drop empty columns
        non_empty_cols = (theory != 0).any(dim=0)
        kept = theory[:, non_empty_cols]

        if kept.size(1) > 0:
            return kept


def generate_dataset(num_samples: int, N: int, max_clauses: int) -> list[torch.Tensor]:
    """
    Generate a list of variable-length consistent theories.
    """
    return [generate_consistent_theory(N, max_clauses) for _ in range(num_samples)]


class TheoryDataset(Dataset):
    def __init__(self, data: list[torch.Tensor]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def theory_collate_fn(batch: list[torch.Tensor]):
    """
    Pad variable-length theories in a batch to the local max clause length.
    """
    batch_size = len(batch)
    N = batch[0].size(0)
    max_m = max(theory.size(1) for theory in batch)

    x_0 = torch.zeros((batch_size, N, max_m), dtype=torch.long)
    clause_mask = torch.zeros((batch_size, max_m), dtype=torch.bool)

    for b, theory in enumerate(batch):
        m_i = theory.size(1)
        x_0[b, :, :m_i] = theory
        clause_mask[b, :m_i] = True

    return x_0, clause_mask


def get_dataloaders(num_samples: int, N: int, max_clauses: int,
                    batch_size: int, train_ratio: float = 0.8):
    data = generate_dataset(num_samples, N, max_clauses)
    dataset = TheoryDataset(data)

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=theory_collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=theory_collate_fn)

    return train_loader, test_loader, data


def resolve_curriculum_max_clauses(epoch: int, stages: list[tuple[int, int]], default_max: int) -> int:
    active = default_max
    for start_epoch, stage_max in stages:
        if epoch >= start_epoch:
            active = stage_max
        else:
            break
    return active


def get_uniform_transition_matrix(beta, num_classes=3):
    Q_t = (1 - beta * 3 / 2) * torch.eye(num_classes) + (beta / 2) * torch.ones((num_classes, num_classes))
    return Q_t


class D3PMForwardCorruption(nn.Module):
    def __init__(self, num_classes=3, num_timesteps=1000):
        super().__init__()
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        scale = 1000 / num_timesteps
        betas = torch.linspace(scale * 1e-4, scale * 0.02, num_timesteps)

        Q_one_step_mats = []
        bar_Q_t = torch.eye(num_classes)
        Q_bars = []

        for beta in betas:
            Q_t = get_uniform_transition_matrix(beta)
            Q_one_step_mats.append(Q_t)
            bar_Q_t = torch.matmul(bar_Q_t, Q_t)
            Q_bars.append(bar_Q_t)

        self.register_buffer('Q_one_step_mats', torch.stack(Q_one_step_mats))
        self.register_buffer('Q_bar_mats', torch.stack(Q_bars))

    def q_sample(self, x_0, t, clause_mask=None):
        batch_size, N, M = x_0.shape
        bar_Q_t = self.Q_bar_mats[t]

        x_0_one_hot = F.one_hot(x_0, num_classes=self.num_classes).float()
        x_0_flat = x_0_one_hot.view(batch_size, N * M, self.num_classes)
        probs_flat = torch.bmm(x_0_flat, bar_Q_t)

        probs = probs_flat.view(batch_size, N, M, self.num_classes)
        x_t_flat = torch.multinomial(probs.view(-1, self.num_classes), num_samples=1).squeeze(-1)
        x_t = x_t_flat.view(batch_size, N, M)

        if clause_mask is not None:
            x_t = x_t.masked_fill(~clause_mask.unsqueeze(1), 0)

        return x_t

    def compute_batch_stats(self, model, x_0, clause_mask):
        batch_size = x_0.size(0)
        device = x_0.device

        # Sample t uniformly from [1, T).
        # t=0 means x_t == x_0 (no noise), so the model trivially memorises — excluded.
        t = torch.randint(1, self.num_timesteps, (batch_size,), device=device).long()
        x_t = self.q_sample(x_0, t, clause_mask=clause_mask)
        predicted_logits = model(x_t, t, clause_mask=clause_mask)

        ce_per_cell = F.cross_entropy(
            predicted_logits.permute(0, 3, 1, 2),
            x_0,
            reduction='none'
        )
        valid_cell_mask = clause_mask.unsqueeze(1).expand_as(ce_per_cell).float()
        loss = (ce_per_cell * valid_cell_mask).sum() / valid_cell_mask.sum().clamp_min(1.0)

        predicted_x0 = predicted_logits.argmax(dim=-1)

        consistent_count = 0
        for b in range(batch_size):
            active_theory = predicted_x0[b, :, clause_mask[b]].detach().cpu()
            if active_theory.numel() > 0 and is_consistent(active_theory):
                consistent_count += 1

        return {
            "loss": loss,
            "predicted_x0": predicted_x0,
            "consistent_count": consistent_count,
            "batch_size": batch_size,
        }

    def forward(self, model, x_0, clause_mask):
        return self.compute_batch_stats(model, x_0, clause_mask)["loss"]


# ==========================================
# Neural Network Architecture (The Denoiser)
# ==========================================
class TheoryDenoiserNet(nn.Module):
    def __init__(self, N, M, num_classes=3, d_model=32, num_timesteps=1000):
        super().__init__()
        self.N = N
        self.M = M
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.d_model = d_model

        # Embed each discrete state (0, 1, 2)
        self.state_embedding = nn.Embedding(num_classes, d_model)

        # PROJECTION LAYER: Takes all N literals in a clause and fuses them into one token
        self.clause_proj = nn.Linear(N * d_model, d_model)

        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Transformer now treats each clause (column) as a single token. Sequence length is M.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Project the output clause token back into N separate literal predictions
        self.output_projection = nn.Linear(d_model, N * num_classes)

    def forward(self, x_t, t, clause_mask=None):
        batch_size, N, M = x_t.size()

        # 1. Embed states: (Batch, N, M) -> (Batch, N, M, d_model)
        x_emb = self.state_embedding(x_t)

        # 2. Group by clause: Rearrange to (Batch, M, N, d_model)
        x_emb = x_emb.permute(0, 2, 1, 3)

        # Flatten the N literals into the feature dimension: (Batch, M, N * d_model)
        x_emb = x_emb.reshape(batch_size, M, N * self.d_model)

        # Project to d_model: (Batch, M, d_model)
        x_seq = self.clause_proj(x_emb)

        # Normalize time and add embedding
        t_normalized = t.float() / self.num_timesteps
        t_emb = self.time_embed(t_normalized.unsqueeze(-1)).view(batch_size, 1, -1)
        x_seq = x_seq + t_emb

        # PyTorch Transformer padding mask (True means "ignore this position")
        src_key_padding_mask = ~clause_mask if clause_mask is not None else None

        # Transform! Sequence length is strictly M.
        encoded = self.transformer(x_seq, src_key_padding_mask=src_key_padding_mask)

        # Predict: (Batch, M, N * num_classes)
        logits = self.output_projection(encoded)

        # Reshape back to (Batch, N, M, num_classes)
        logits = logits.view(batch_size, M, N, self.num_classes).permute(0, 2, 1, 3)

        return logits


def sample_theories(model, corrupt, N, M, batch_size, num_timesteps, device, clause_mask=None):
    """
    Generate theories from pure noise using the reverse diffusion process with Re-Noising.
    """
    model.eval()
    with torch.no_grad():
        # Start with pure noise
        x_t = torch.randint(0, 3, (batch_size, N, M), device=device)

        # Ensure padding positions remain masked out with '0' (absent)
        if clause_mask is not None:
            x_t = x_t.masked_fill(~clause_mask.unsqueeze(1), 0)

        for t_step in reversed(range(num_timesteps)):
            t_tensor = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
            logits = model(x_t, t_tensor, clause_mask=clause_mask)

            # Model predicts the CLEAN theory (x_0)
            x_0_pred = logits.argmax(dim=-1)

            if t_step > 0:
                # RE-NOISING TRICK: Add t-1 noise back onto the clean prediction
                t_minus_1 = torch.full((batch_size,), t_step - 1, device=device, dtype=torch.long)
                x_t = corrupt.q_sample(x_0_pred, t_minus_1, clause_mask=clause_mask)
            else:
                x_t = x_0_pred

    return x_t


def run_epoch(model, corrupt, dataloader, optimizer=None):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    consistent_count = 0
    total_count = 0

    for x_0, clause_mask in dataloader:
        if is_training:
            optimizer.zero_grad()
            batch_stats = corrupt.compute_batch_stats(model, x_0, clause_mask)
            loss = batch_stats["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # For training, we can keep the fast 1-shot metric just to monitor progress
            consistent_count += batch_stats["consistent_count"]
            total_count += batch_stats["batch_size"]
        else:
            with torch.no_grad():
                # 1. Compute Validation Loss using normal forward stats
                batch_stats = corrupt.compute_batch_stats(model, x_0, clause_mask)
                loss = batch_stats["loss"]
                total_loss += loss.item()

                # 2. Evaluate true generation quality iteratively using Re-Noising
                batch_size = x_0.size(0)
                device = x_0.device
                M_batch = x_0.size(2)

                generated_theories = sample_theories(
                    model=model,
                    corrupt=corrupt,
                    N=model.N,
                    M=M_batch,
                    batch_size=batch_size,
                    num_timesteps=model.num_timesteps,
                    device=device,
                    clause_mask=clause_mask
                )

                # 3. Check SAT Consistency of generated outputs
                batch_consistent = 0
                for b in range(batch_size):
                    active_theory = generated_theories[b, :, clause_mask[b]].detach().cpu()
                    if active_theory.numel() > 0 and is_consistent(active_theory):
                        batch_consistent += 1

                consistent_count += batch_consistent
                total_count += batch_size

    avg_loss = total_loss / len(dataloader)
    consistency_rate = 100.0 * consistent_count / total_count if total_count else 0.0
    return avg_loss, consistent_count, total_count, consistency_rate


# ==========================================
# Example Training Loop
# ==========================================
if __name__ == "__main__":
    # ── Hyperparameters ────────────────────────────────────────────────────────
    N_LITERALS = 5
    M_CLAUSES = 20  # max number of clauses per theory
    K_STATES = 3
    NUM_SAMPLES = 1000
    BATCH_SIZE = 16
    TRAIN_RATIO = 0.8
    EPOCHS = 100
    LR = 1e-4
    NUM_TIMESTEPS = 1000
    SANITY_CHECK_LIMIT = 200

    # (start_epoch, max_clauses). Theory size grows with training.
    CURRICULUM_STAGES = [
        (1, max(1, M_CLAUSES // 3)),
        (max(2, EPOCHS // 3), max(1, (2 * M_CLAUSES) // 3)),
        (max(3, (2 * EPOCHS) // 3), M_CLAUSES),
    ]

    log_lines: list[str] = []


    def emit(msg: str) -> None:
        print(msg)
        log_lines.append(msg)


    # ── Build initial dataset & loaders ───────────────────────────────────────
    active_max_clauses = resolve_curriculum_max_clauses(1, CURRICULUM_STAGES, M_CLAUSES)
    emit(
        f"Generating {NUM_SAMPLES} consistent theories "
        f"(N={N_LITERALS}, active max M={active_max_clauses}, global max M={M_CLAUSES})..."
    )
    train_loader, test_loader, all_data = get_dataloaders(
        num_samples=NUM_SAMPLES,
        N=N_LITERALS,
        max_clauses=active_max_clauses,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
    )
    train_count = int(NUM_SAMPLES * TRAIN_RATIO)
    test_count = NUM_SAMPLES - train_count
    emit(f"  Train batches : {len(train_loader)}  ({train_count} theories)")
    emit(f"  Test  batches : {len(test_loader)}  ({test_count} theories)")

    sanity_subset = all_data[:min(SANITY_CHECK_LIMIT, len(all_data))]
    n_empty = sum(1 for th in sanity_subset if th.size(1) == 0)
    n_inconsistent = sum(1 for th in sanity_subset if not is_consistent(th))
    emit(f"  Empty theories in sanity subset: {n_empty}  (should be 0)")
    emit(f"  Inconsistent theories in sanity subset: {n_inconsistent}  (should be 0)")
    emit(f"  Sanity subset size: {len(sanity_subset)} / {len(all_data)}\n")

    # ── Model, optimizer ───────────────────────────────────────────────────────
    corrupt = D3PMForwardCorruption(num_classes=K_STATES, num_timesteps=NUM_TIMESTEPS)
    model = TheoryDenoiserNet(N=N_LITERALS, M=M_CLAUSES, num_classes=K_STATES, num_timesteps=NUM_TIMESTEPS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Training + Evaluation loop ─────────────────────────────────────────────
    current_stage_max = active_max_clauses

    for epoch in range(1, EPOCHS + 1):
        stage_max = resolve_curriculum_max_clauses(epoch, CURRICULUM_STAGES, M_CLAUSES)

        if stage_max != current_stage_max:
            current_stage_max = stage_max
            emit(f"\n[Curriculum] Epoch {epoch}: switching active max clauses to {current_stage_max}")
            train_loader, test_loader, all_data = get_dataloaders(
                num_samples=NUM_SAMPLES,
                N=N_LITERALS,
                max_clauses=current_stage_max,
                batch_size=BATCH_SIZE,
                train_ratio=TRAIN_RATIO,
            )
            emit(f"  Train batches : {len(train_loader)}  ({train_count} theories)")
            emit(f"  Test  batches : {len(test_loader)}  ({test_count} theories)")

        train_loss, train_consistent, train_total, train_consistency = run_epoch(
            model=model,
            corrupt=corrupt,
            dataloader=train_loader,
            optimizer=optimizer,
        )

        test_loss, test_consistent, test_total, test_consistency = run_epoch(
            model=model,
            corrupt=corrupt,
            dataloader=test_loader,
            optimizer=None,
        )

        emit(
            f"Epoch {epoch:>3}/{EPOCHS} | "
            f"Stage max M: {current_stage_max} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Consistent: {train_consistent}/{train_total} ({train_consistency:.1f}%) | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Consistent: {test_consistent}/{test_total} ({test_consistency:.1f}%)"
        )

    # ── Persist artifacts ──────────────────────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"theory_denoiser_{run_id}.pt"
    log_path = output_dir / f"training_log_{run_id}.txt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "N_LITERALS": N_LITERALS,
                "M_CLAUSES": M_CLAUSES,
                "K_STATES": K_STATES,
                "NUM_SAMPLES": NUM_SAMPLES,
                "BATCH_SIZE": BATCH_SIZE,
                "TRAIN_RATIO": TRAIN_RATIO,
                "EPOCHS": EPOCHS,
                "LR": LR,
                "NUM_TIMESTEPS": NUM_TIMESTEPS,
                "SAT_SOLVER_BACKEND": SAT_SOLVER_BACKEND,
                "CURRICULUM_STAGES": CURRICULUM_STAGES,
            },
        },
        model_path,
    )

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    emit(f"Saved model checkpoint: {model_path}")
    emit(f"Saved training log: {log_path}")