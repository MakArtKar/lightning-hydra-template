# AGENTS.md — Plan & Act Runbook

Short, actionable runbook for agents collaborating in this repository.

## Pre-flight questions

Before starting any non-trivial task, answer these questions:

- **Goal**: What outcome is required and why?
- **Scope**: Which files/modules/configs will be changed? What is out-of-scope?
- **Acceptance criteria**: Verifiable checks (tests passing, behavior visible, docs updated)
- **Constraints**: Performance, security, compatibility (Hydra/Lightning), environment (`.venv`)
- **Risks & rollbacks**: Identify risky edits, propose a safe fallback

## Execution loop

### 1. Prepare environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

### 2. Specification-first for complex tasks
- Write a brief SPEC (problem, approach, acceptance) using `docs/SPEC_TEMPLATE.md`
- Get approval before large code changes

### 3. Edit in small, verifiable steps
- Keep edits scoped; prefer isolated modules and minimal diffs
- One logical change per commit

### 4. Local checks after each step
```bash
make format      # runs pre-commit on all files
make test        # fast tests (skips slow tests)
make test-full   # all tests including slow ones
```

### 5. Functional checks (when relevant)
```bash
python ml_core/train.py --config-dir configs debug=fdr
```
**Note**: Always include `--config-dir configs` when invoking `ml_core/train.py` or `ml_core/eval.py` from the repo root. If running from inside `ml_core/`, use `--config-dir ../configs`.

### 6. Before PR: run local CI (optional but recommended)
```bash
make ci-local         # run all CI jobs locally
make ci-tests-ubuntu  # run ubuntu test suite
make ci-codequality-pr # run pre-commit checks
```

### 7. Summarize impact
- What changed, why it's safe, how to validate

## Pull Request checklist

- [ ] Green CI: all tests and code-quality workflows pass
- [ ] Local CI validated with `act` (optional but recommended)
- [ ] Docs updated where applicable (README, configs, ADRs for notable decisions)
- [ ] No skipped security checks; no secrets committed
- [ ] Logs do not contain sensitive data
- [ ] Concise PR description with changes, impact, and verification commands

## Local CI (act) quick start

`act` runs GitHub Actions workflows locally in Docker containers.

### Installation

**macOS**:
```bash
brew tap nektos/tap && brew install nektos/tap/act
# Ensure Docker Desktop is running
```

**Ubuntu**:
```bash
curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
# or: sudo snap install act
```

### Available commands

```bash
make ci-list-tests       # List all jobs in test workflow
make ci-tests-ubuntu     # Run Ubuntu matrix job
make ci-tests-macos      # Run macOS matrix job (mapped to Ubuntu image)
make ci-tests-windows    # Run Windows matrix job (mapped to Ubuntu image)
make ci-codequality-pr   # Run pre-commit on changed files
make ci-codequality-main # Run pre-commit on all files
make ci-coverage         # Run coverage job (Codecov upload may no-op locally)
```

## Context rules

- Prefer exact references to files, functions, and diffs.
- Use minimal, runnable commands; avoid large log dumps.
- Use tags in Hydra runs when helpful: `python ml_core/train.py --config-dir configs tags=["dev"]`.

## Context Kits

- Training loop tweaks

  - Files to open:
    - `ml_core/train.py`
    - `ml_core/models/base_module.py`
    - `configs/model/*.yaml`
    - `configs/trainer/*.yaml`
  - Sample ripgrep searches:
    - `rg -n "training_step|validation_step|test_step" ml_core/models`
    - `rg -n "configure_optimizers|optimizer|lr_scheduler" ml_core/models`
    - `rg -n "Trainer|fit\\(" ml_core/train.py`
  - Quick debug run:
    - `python -m ml_core.train --config-dir configs experiment=example debug=fdr`

- Data pipeline changes

  - Files to open:
    - `ml_core/data/`
    - `ml_core/transforms/`
    - `configs/data/*.yaml`
  - Sample ripgrep searches:
    - `rg -n "DataModule|setup|prepare_data" ml_core/data`
    - `rg -n "train_dataloader|val_dataloader|test_dataloader" ml_core/data`
    - `rg -n "transform|Compose|Normalize" ml_core/transforms`
    - `rg -n "batch_size|num_workers" configs/data configs/trainer`
  - Quick debug run:
    - `python -m ml_core.train --config-dir configs experiment=example debug=fdr`

- Metrics/loss changes

  - Files to open:
    - `ml_core/models/utils.py`
    - `ml_core/models/components/`
  - Sample ripgrep searches:
    - `rg -n "loss" ml_core/models`
    - `rg -n "metric|Accuracy|F1|Precision|Recall" ml_core/models`
    - `rg -n "log\\(" ml_core/models`
  - Quick debug run:
    - `python -m ml_core.train --config-dir configs experiment=example debug=fdr`

- Debug runs

  - See `docs/DEBUG.md` for details and additional profiles.
  - Fast dev run command:
    - `python -m ml_core.train --config-dir configs experiment=example debug=fdr`

## References

- `docs/SPEC_TEMPLATE.md` — Template for writing specifications
- `docs/DEBUG.md` — Debug profiles and quick testing
- `docs/TESTING.md` — Testing strategy and commands
- `docs/SECURITY.md` — Security policies and best practices
- `docs/ARCHITECTURE.md` — System architecture overview
- `docs/HOW_TO_IMPLEMENT_THE_TASK.md` — Multi-agent task workflow

## Quick verification

```bash
# Verify documentation exists
test -f AGENTS.md && test -f CLAUDE.md && echo "Documentation: OK"
```
