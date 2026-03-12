# DiscreteDiffusion

A compact PyTorch prototype for **discrete diffusion over CNF-style theories** stored as an `N x M` matrix.

- `N` = number of literals
- `M` = number of clauses
- each cell is one categorical value:
  - `0` = literal absent from that clause
  - `1` = positive literal
  - `2` = negative literal

In the matrix representation used by `main.py`:
- each **column** is one CNF clause (a disjunction)
- each **row** tracks one literal across clauses
- the whole theory is the conjunction of all columns

## What the script does

When you run `python main.py`, the script:

1. generates a dataset of theories that are consistent by construction
2. splits them into train and test sets
3. verifies consistency with a **SAT solver**
4. applies discrete forward corruption with precomputed transition matrices
5. trains a Transformer denoiser to predict the clean theory from noisy samples
6. prints train/test loss per epoch

## SAT-based consistency check

`is_consistent(theory)` now uses **PySAT** instead of a naive polarity check.

### Encoding used

For a theory matrix of shape `(N, M)`:
- variable `i` in SAT becomes integer `i + 1`
- if `theory[i, j] == 1`, clause `j` contains literal `+(i + 1)`
- if `theory[i, j] == 2`, clause `j` contains literal `-(i + 1)`
- if `theory[i, j] == 0`, literal `i` is omitted from clause `j`

Example:

```text
Theory matrix:
[[1, 0],
 [2, 1]]

Column 0 -> [1, -2]    meaning (x1 ∨ ¬x2)
Column 1 -> [2]        meaning (x2)
CNF      -> (x1 ∨ ¬x2) ∧ (x2)
```

### Edge cases

- **No clauses** (`M = 0`) -> satisfiable
- **All-zero column** -> empty clause -> unsatisfiable
- **Tautological clause** containing both `x` and `¬x` is skipped defensively

## Main components

### `theory_to_cnf_clauses(theory)`
Converts one `(N, M)` theory matrix into a list of SAT clauses in PySAT format.

### `is_consistent(theory)`
Calls the SAT solver and returns `True` iff the CNF theory is satisfiable.

### `generate_consistent_theory(N, M)`
Creates a theory that is consistent by construction by restricting each literal to one polarity family:
- absent-only
- absent/positive
- absent/negative

### `D3PMForwardCorruption`
Owns the discrete diffusion forward process:
- builds `Q_t`
- precomputes cumulative `Q̄_t`
- samples noisy `x_t`
- computes the denoising training loss

### `TheoryDenoiserNet`
A Transformer-based denoiser that:
- embeds discrete states
- embeds timestep `t`
- runs self-attention over the flattened `N*M` grid
- predicts per-cell logits for the clean theory

## Installation

Create/activate your virtual environment, then install:

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

Expected output is a dataset summary followed by epoch losses, for example:

```text
Generating 1000 consistent theories (4x4)...
  Train batches : 50  (800 theories)
  Test  batches : 13  (200 theories)
  Inconsistent theories in dataset: 0  (should be 0)

Epoch   1/10 | Train Loss: 0.91 | Test  Loss: 0.88
...
```

## Notes

- The transition matrix helper is currently written for the 3-state setup.
- The generated dataset is synthetic and mainly useful for validating the pipeline.
- SAT checking is used for correctness of the consistency definition, even though the generator already produces satisfiable theories.
