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

    This is the semantic notion of consistency:
      - each column is a disjunction (clause)
      - the whole matrix is the conjunction of those clauses
      - the theory is consistent iff there exists a truth assignment that
        satisfies all clauses simultaneously

    Edge cases:
      - M = 0 (no clauses) -> satisfiable / consistent
      - an all-zero column encodes the empty clause -> unsatisfiable
    """
    clauses = theory_to_cnf_clauses(theory)

    with Solver(name=SAT_SOLVER_BACKEND) as solver:
        solver.append_formula(clauses)
        return solver.solve()


def generate_consistent_theory(N: int, max_clauses: int) -> torch.Tensor:
    """
    Generate one consistent theory with variable clause length.

    Returns a tensor of shape (N, K), where 1 <= K <= max_clauses.
    Empty clauses are dropped; if all clauses are dropped, generation retries.
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
                theory[i] = torch.randint(0, 2, (max_clauses,))      # {0,1}
            else:
                theory[i] = torch.randint(0, 2, (max_clauses,)) * 2  # {0,2}

        non_empty_cols = (theory != 0).any(dim=0)
        kept = theory[:, non_empty_cols]  # drop empty-clause columns

        if kept.size(1) > 0:
            return kept


def generate_dataset(num_samples: int, N: int, max_clauses: int) -> list[torch.Tensor]:
    """
    Generate a list of variable-length consistent theories.

    Each element has shape (N, K_i), where 1 <= K_i <= max_clauses.
    """
    return [generate_consistent_theory(N, max_clauses) for _ in range(num_samples)]


class TheoryDataset(Dataset):
    """
    A Dataset of variable-length theories (N x K_i).
    """
    def __init__(self, data: list[torch.Tensor]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def theory_collate_fn(batch: list[torch.Tensor]):
    """
    Pad variable-length theories in a batch to the local max clause length.

    Returns:
      x_0: (B, N, M_batch) padded with zeros
      clause_mask: (B, M_batch) True for real clauses, False for padding
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
    """
    Build train and test loaders for variable-length theories.

    Returns:
        train_loader, test_loader, data
    """
    data = generate_dataset(num_samples, N, max_clauses)
    dataset = TheoryDataset(data)

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=theory_collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=theory_collate_fn)

    return train_loader, test_loader, data


def resolve_curriculum_max_clauses(epoch: int, stages: list[tuple[int, int]], default_max: int) -> int:
    """
    Resolve active max_clauses for a given epoch from a stage list.

    Args:
        epoch: 1-based epoch index.
        stages: list of (start_epoch, max_clauses), sorted by start_epoch.
        default_max: fallback max_clauses if no stage matches.
    """
    active = default_max
    for start_epoch, stage_max in stages:
        if epoch >= start_epoch:
            active = stage_max
        else:
            break
    return active


def get_uniform_transition_matrix(beta, num_classes=3):
    """
    Creates a uniform transition matrix Q_t for a given beta.
    Diagonal gets (1 - beta + beta/K), off-diagonals get (beta/K).
    """
    Q_t = (1 - beta * 3/2) * torch.eye(num_classes) + (beta / 2) * torch.ones((num_classes, num_classes))
    return Q_t


class D3PMForwardCorruption(nn.Module):
    def __init__(self, num_classes=3, num_timesteps=1000):
        super().__init__()
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        # 1. Define the beta noise schedule (linear schedule)
        # We scale beta to ensure constant cumulative noise injection
        scale = 1000 / num_timesteps
        betas = torch.linspace(scale * 1e-4, scale * 0.02, num_timesteps)

        # 2. Precompute transition matrices Q_t and cumulative matrices Q_bar_t
        Q_one_step_mats = []
        bar_Q_t = torch.eye(num_classes)
        Q_bars = []

        for beta in betas:
            Q_t = get_uniform_transition_matrix(beta)
            Q_one_step_mats.append(Q_t)

            # \bar{Q}_t = Q_1 * Q_2 * ... * Q_t
            bar_Q_t = torch.matmul(bar_Q_t, Q_t)
            Q_bars.append(bar_Q_t)

        self.register_buffer('Q_one_step_mats', torch.stack(Q_one_step_mats))
        self.register_buffer('Q_bar_mats', torch.stack(Q_bars))

    def q_sample(self, x_0, t, clause_mask=None):
        """
        The Forward Process: Sample noisy theory x_t ~ q(x_t | x_0)

        Args:
            x_0: Clean, consistent theories. Shape (batch_size, N, M). Values in {0, 1, 2}.
            t: Timesteps for the batch. Shape (batch_size,).
            clause_mask: Optional mask of shape (batch_size, M), True for real clauses.
        Returns:
            x_t: Noisy theories with injected MUCs. Shape (batch_size, N, M).
        """
        # batch_size: Number of theories in the batch
        # N: Number of literals per theory
        # M: Max number of clauses per theory
        batch_size, N, M = x_0.shape

        # Fetch the cumulative transition matrix for the specific timesteps t
        # Shape: (batch_size, K, K)
        bar_Q_t = self.Q_bar_mats[t]

        # Convert x_0 to one-hot encoding to perform categorical matrix multiplication
        # Shape: (batch_size, N, M, K)
        x_0_one_hot = F.one_hot(x_0, num_classes=self.num_classes).float() # return one hot encoding per element in x_0

        # Flatten spatial dimensions to perform batched matrix multiplication
        # (batch_size, N*M, K) @ (batch_size, K, K) -> (batch_size, N*M, K)
        x_0_flat = x_0_one_hot.view(batch_size, N * M, self.num_classes)
        probs_flat = torch.bmm(x_0_flat, bar_Q_t)

        # Reshape back to theory matrix dimensions
        probs = probs_flat.view(batch_size, N, M, self.num_classes)

        # Sample x_t from the resulting categorical distributions
        # We flatten the batch and spatial dims to use torch.multinomial
        x_t_flat = torch.multinomial(probs.view(-1, self.num_classes), num_samples=1).squeeze(-1)
        x_t = x_t_flat.view(batch_size, N, M)

        # Keep padded columns inactive; otherwise padding noise pollutes attention.
        if clause_mask is not None:
            x_t = x_t.masked_fill(~clause_mask.unsqueeze(1), 0)

        return x_t

    def compute_batch_stats(self, model, x_0, clause_mask):
        """
        Run one denoising batch and return loss and consistency stats.

        Args:
            x_0: (B, N, M_batch)
            clause_mask: (B, M_batch) True for real clauses
        """
        batch_size = x_0.size(0)
        device = x_0.device

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        x_t = self.q_sample(x_0, t, clause_mask=clause_mask)
        predicted_logits = model(x_t, t, clause_mask=clause_mask)

        # Masked CE so padded columns do not contribute to optimization.
        ce_per_cell = F.cross_entropy(
            predicted_logits.permute(0, 3, 1, 2),
            x_0,
            reduction='none'
        )  # (B, N, M_batch)
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
    """
    A lightweight Transformer-based architecture to denoise the N x M logic theory.
    Treats the clauses (M) as a sequence, allowing the model to naturally handle
    permutation invariance of logic rules.
    """

    def __init__(self, N, M, num_classes=3, d_model=128, num_timesteps=1000):
        super().__init__()
        self.N = N
        self.M = M
        self.num_timesteps = num_timesteps

        # Embed the discrete states (0, 1, 2) into continuous vectors
        self.state_embedding = nn.Embedding(num_classes, d_model)

        # FIX 1: Add Row Embeddings to provide spatial awareness for literals (which are NOT permutation invariant)
        self.row_embedding = nn.Embedding(N, d_model)

        # Embed the timestep t
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # A simple Transformer Encoder to pass messages between clauses
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Project back to the categorical vocabulary for each literal
        self.output_projection = nn.Linear(d_model, num_classes)

    def forward(self, x_t, t, clause_mask=None):
        batch_size, N, M = x_t.size()

        # Generate and apply row embeddings
        row_idx = torch.arange(self.N, device=x_t.device).view(1, self.N, 1)
        r_emb = self.row_embedding(row_idx)

        x_emb = self.state_embedding(x_t)

        # FIX 2: Normalize timesteps so time embeddings don't overpower state embeddings
        t_normalized = t.float() / self.num_timesteps
        t_emb = self.time_embed(t_normalized.unsqueeze(-1)).view(batch_size, 1, 1, -1)

        # Inject both time and row awareness
        x_emb = x_emb + t_emb + r_emb

        x_seq = x_emb.view(batch_size, N * M, -1)

        src_key_padding_mask = None
        if clause_mask is not None:
            token_mask = clause_mask.unsqueeze(1).expand(batch_size, N, M).reshape(batch_size, N * M)
            src_key_padding_mask = ~token_mask

        encoded = self.transformer(x_seq, src_key_padding_mask=src_key_padding_mask)
        logits = self.output_projection(encoded)

        return logits.view(batch_size, N, M, -1)


# FIX 3: Iterative reverse-process sampling function
def sample_theories(model, N, M, batch_size, num_timesteps, device):
    """
    Generate theories from pure noise using the reverse diffusion process (step by step).
    """
    model.eval()
    with torch.no_grad():
        x_t = torch.randint(0, 3, (batch_size, N, M), device=device)
        for t_step in reversed(range(num_timesteps)):
            t_tensor = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
            logits = model(x_t, t_tensor)
            x_t = logits.argmax(dim=-1)
    return x_t

def run_epoch(model, corrupt, dataloader, optimizer=None):
    """
    Run one full epoch.

    Args:
        model: Denoising model.
        corrupt: Diffusion/corruption module.
        dataloader: Train or test DataLoader.
        optimizer: Optimizer for training. If None, runs evaluation only.

    Returns:
        avg_loss: Average batch loss across the epoch.
        consistent_count: Number of SAT-consistent denoised theories.
        total_count: Total number of denoised theories evaluated.
        consistency_rate: Percentage of denoised theories that are SAT-consistent.
    """
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
            consistent_count += batch_stats["consistent_count"]
            total_count += batch_stats["batch_size"]
        else:
            with torch.no_grad():
                # Compute loss using the standard noisy forward pass
                batch_stats = corrupt.compute_batch_stats(model, x_0, clause_mask)
                loss = batch_stats["loss"]
                total_loss += loss.item()

                # FIX 3: Evaluate true generation quality iteratively (rather than 1-shot denoising)
                batch_size = x_0.size(0)
                device = x_0.device
                generated_theories = sample_theories(model, model.N, x_0.size(2), batch_size, model.num_timesteps,
                                                     device)

                batch_consistent = 0
                for b in range(batch_size):
                    # We evaluate on the valid (unpadded) clauses length for each sample
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
    M_CLAUSES = 20         # max number of clauses per theory
    K_STATES = 3
    NUM_SAMPLES = 1000
    BATCH_SIZE = 16
    TRAIN_RATIO = 0.8
    EPOCHS = 1000
    LR = 1e-4
    NUM_TIMESTEPS = 1000
    SANITY_CHECK_LIMIT = 200

    # (start_epoch, max_clauses). Theory size grows with training.
    CURRICULUM_STAGES = [
        (1, max(1, M_CLAUSES // 3)),
        (max(2, EPOCHS // 3), max(1, (2 * M_CLAUSES) // 3)),
        (max(3, (2 * EPOCHS) // 3), M_CLAUSES),
    ]

    # Track all printed messages so we can persist an exact training log.
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

    # Bounded startup sanity check to avoid too many native SAT calls.
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

        # Rebuild data only when curriculum stage changes.
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
