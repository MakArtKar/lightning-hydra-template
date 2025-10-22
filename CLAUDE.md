# CLAUDE.md — Concise Collaboration Rules

High-signal, compact checklist for AI-assisted development in this repo.

## Plan & Act (use for any nontrivial task)

- State goal and context succinctly: what, why, where (files/paths).
- Propose a short plan with risks: steps, touched files, tests impacted.
- Stress-test the plan: edge cases, performance, security, data integrity.
- Freeze scope: acceptance criteria, out-of-scope, success check.
- Decompose: create small, verifiable todos with clear outcomes.
- Execute with checks: after each step, run format and tests; post a brief status update.

## Lean context

- Point to exact files/functions and prefer diffs over prose.
- Include minimal, runnable commands; avoid dumping long logs.
- Prefer code references and tight excerpts over full files.

## Local workflow (.venv required)

- Always use a local virtualenv at `.venv` in repo root:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # macOS/zsh
  pip install -r requirements.txt
  pre-commit install
  ```
- Run format and tests locally before edits and before pushing:
  ```bash
  make format           # runs pre-commit on all files
  make test             # pytest -k "not slow"
  make test-full        # pytest (all tests)
  # before PR: run local CI mirror via act (Ubuntu-only)
  make ci-local
  # run macOS/Windows matrix jobs locally (mapped to Ubuntu images)
  make ci-tests-macos
  make ci-tests-windows
  ```

## act installation

- macOS:
  ```bash
  brew tap nektos/tap && brew install nektos/tap/act
  ```
- Ubuntu:
  ```bash
  curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
  # or: sudo snap install act
  ```
- Useful training/debug commands:
  ```bash
  python ml_core/train.py                 # default
  python ml_core/train.py debug=default   # debug profile
  python ml_core/train.py debug=fdr       # 1 batch train/val/test
  python ml_core/train.py debug=profiler  # profile timings
  python ml_core/train.py debug=overfit   # try overfitting to 1 batch
  ```

## Tests are specs

- If a test fails, fix the code/config first. Do not change tests without explicit approval.
- Keep tests fast by default; use `-k "not slow"` locally unless you need full coverage.

## Security hygiene

- Never commit secrets; use `.env` for local values (excluded by git).
- Respect scanners and hooks; keep changes minimal in scope.
- Avoid printing sensitive data in logs; sanitize debugging output.

## Change discipline

- Keep edits minimal and isolated; explain rationale in the PR description.
- Reference the exact configs/flags changed (e.g., `trainer.limit_*`, logger settings).

## References (future docs to be added)

- `docs/DEBUG.md` — Hydra debug profiles and tips
- `docs/SPEC_TEMPLATE.md` — spec-before-code template
- `docs/ADR_TEMPLATE.md` — architecture decision records

______________________________________________________________________

Quick verification:

```bash
make format && make test
```
