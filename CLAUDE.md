# CLAUDE.md — Concise Collaboration Rules

High-signal, compact checklist for AI-assisted development in this repo.

## Plan & Act (use for any non-trivial task)

1. **State goal and context**: What, why, where (files/paths)
2. **Propose a plan**: Steps, touched files, tests impacted, potential risks
3. **Stress-test**: Edge cases, performance, security, data integrity
4. **Freeze scope**: Define acceptance criteria, out-of-scope items, success checks
5. **Decompose**: Create small, verifiable todos with clear outcomes
6. **Execute with checks**: After each step, run format and tests; post brief status

## Lean context

- Point to exact files/functions and prefer diffs over prose.
- Include minimal, runnable commands; avoid dumping long logs.
- Prefer code references and tight excerpts over full files.

## Local workflow (.venv required)

Always use a local virtualenv at `.venv` in repo root:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
pre-commit install
```

Run format and tests locally before pushing:

```bash
make format           # runs pre-commit on all files
make test             # pytest -k "not slow"
make test-full        # pytest (all tests)
```

Optional: run local CI mirror before PR:

```bash
make ci-local         # all CI jobs locally (requires Docker)
make ci-tests-ubuntu  # Ubuntu test matrix
make ci-tests-macos   # macOS test matrix (mapped to Ubuntu image)
make ci-tests-windows # Windows test matrix (mapped to Ubuntu image)
```

## act installation

Install `act` to run GitHub Actions locally (requires Docker):

**macOS**:
```bash
brew tap nektos/tap && brew install nektos/tap/act
```

**Ubuntu**:
```bash
curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
# or: sudo snap install act
```

## Useful training/debug commands

```bash
python ml_core/train.py --config-dir configs                 # default run
python ml_core/train.py --config-dir configs debug=default   # debug profile
python ml_core/train.py --config-dir configs debug=fdr       # fast dev run (1 batch)
python ml_core/train.py --config-dir configs debug=profiler  # profile timings
python ml_core/train.py --config-dir configs debug=overfit   # overfit 1 batch
```

## Tests are specs

- **Tests define expected behavior**: If a test fails, fix the code/config first
- **Do not change tests** without explicit approval and justification
- **Keep tests fast**: Use `-k "not slow"` locally unless you need full coverage

## Security hygiene

- Never commit secrets; use `.env` for local values (excluded by git).
- Respect scanners and hooks; keep changes minimal in scope.
- Avoid printing sensitive data in logs; sanitize debugging output.

## Change discipline

- **Minimal edits**: Keep changes isolated and focused
- **Document rationale**: Explain why in the PR description
- **Reference specifics**: Cite exact configs/flags changed (e.g., `trainer.limit_*`, logger settings)
- **One logical change**: Prefer multiple small PRs over one large PR

## References

- `docs/DEBUG.md` — Hydra debug profiles and tips
- `docs/SPEC_TEMPLATE.md` — Specification template
- `docs/TESTING.md` — Testing strategy
- `docs/SECURITY.md` — Security policies
- `docs/ARCHITECTURE.md` — System architecture
- `AGENTS.md` — Detailed agent runbook

## Quick verification

```bash
make format && make test
```
