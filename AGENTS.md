# AGENTS.md — Plan & Act Runbook

Short, actionable runbook for agents collaborating in this repository.

## Pre-flight questions

- Goal: what outcome is required and why?
- Scope: which files/modules/configs will be changed? What is out-of-scope?
- Acceptance criteria: verifiable checks (tests passing, behavior visible, docs updated).
- Constraints: performance, security, compatibility (Hydra/Lightning), environment (`.venv`).
- Risks & rollbacks: identify risky edits, propose a safe fallback.

## Execution loop

1. Prepare `.venv` and deps
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   pre-commit install
   ```
2. Specification-first for complex tasks
   - Write a brief SPEC (problem, approach, acceptance). Link to `docs/SPEC_TEMPLATE.md`.
   - Get approval before large code changes.
3. Edit in small, verifiable steps
   - Keep edits scoped; prefer isolated modules and minimal diffs.
4. Local checks each step
   ```bash
   make format
   make test        # fast tests (not slow)
   # optionally
   make test-full   # all tests
   # before PR: run local CI mirror via act (Ubuntu-only)
   make ci-local
   ```
5. Functional checks (when relevant)
   ```bash
   python ml_core/train.py --config-dir configs debug=fdr
   ```
   Note: Always include `--config-dir configs` when invoking `ml_core/train.py` or `ml_core/eval.py` from the repo root. If running from inside `ml_core/`, use `--config-dir ../configs`.
6. Summarize impact
   - What changed, why it’s safe, how to validate.

## Pull Request checklist

- Green CI: tests and code-quality workflows pass.
- Ran local CI with `act` (Ubuntu-only) and addressed failures.
- Docs updated where applicable (README snippets, configs), ADRs for notable decisions.
- No skipped security checks; no secrets; minimal logging of sensitive data.
- Concise summary of changes, impact, and verification commands.

## Local CI (act) quick start

- macOS install:
  ```bash
  brew tap nektos/tap && brew install nektos/tap/act
  # ensure Docker Desktop is running
  ```
- Ubuntu install:
  ```bash
  curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
  # or: sudo snap install act
  ```
- Run selected jobs locally:
  ```bash
  make ci-list-tests       # list jobs in test workflow
  make ci-tests-ubuntu     # ubuntu matrix job
  make ci-tests-macos      # macOS matrix job (mapped to Ubuntu image)
  make ci-tests-windows    # Windows matrix job (mapped to Ubuntu image)
  make ci-codequality-pr   # pre-commit on changed files
  make ci-codequality-main # pre-commit on all files
  make ci-coverage         # coverage (Codecov upload may no-op locally)
  ```

## Context rules

- Prefer exact references to files, functions, and diffs.
- Use minimal, runnable commands; avoid large log dumps.
- Use tags in Hydra runs when helpful: `python ml_core/train.py --config-dir configs tags=["dev"]`.

## Context kits (stub)

- Will be expanded in Task 16: curated pointers for common workflows (data, models, trainers, loggers).

## References (future docs to be added)

- `docs/SPEC_TEMPLATE.md`
- `docs/ADR_TEMPLATE.md`
- `docs/DEBUG.md`

______________________________________________________________________

Quick verification:

```bash
cat CLAUDE.md AGENTS.md | head -n 40
```
