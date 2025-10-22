# Testing Policy (tests-as-specifications)

This repository treats tests as executable specifications. A change is correct only if it satisfies the existing test suite or explicitly updates the specification with reviewed test changes.

## Principles

- **Tests are specifications**: Behavior is defined by tests. Code changes must pass existing tests. If behavior changes, update tests explicitly in the same PR and call this out in the description.
- **Deterministic and hermetic**: Tests must be repeatable across machines and runs.
  - Seed all randomness where applicable; avoid time- or network-dependent flakiness.
  - No external network or filesystem dependencies unless mocked/faked; prefer `tmp_path` for I/O.
- **Small and focused**: Prefer many small, readable tests over large end-to-end only suites.

## Speed and parallelism

- Use pytest with xdist to parallelize locally:

  ```bash
  pytest -n auto -q
  ```

- Mark genuinely long-running tests with the slow marker and keep them minimal:

  ```python
  import pytest

  @pytest.mark.slow
  def test_expensive_path():
      ...
  ```

- Run only fast tests by default during development:

  ```bash
  pytest -n auto -m "not slow" -q
  ```

Notes:

- `pyproject.toml` enables `--strict-markers` and defines `slow`.
- xdist is recommended; install via `pip install pytest-xdist` if not already available.

## Coverage

- Aim for meaningful coverage of critical logic; avoid chasing 100% without value.

- Recommended local command:

  ```bash
  pytest -n auto --cov=ml_core --cov-report=term-missing
  ```

## Fixtures and patterns

- Prefer built-in fixtures like `tmp_path`, `monkeypatch` and small helper fixtures.
- Repository-provided fixtures in `tests/conftest.py` supply Hydra configs:
  - `cfg_train`/`cfg_train_global`
  - `cfg_eval`/`cfg_eval_global`
- Keep fixtures simple, explicit, and reusable; avoid hidden side effects.

## Workflow

- During development:
  - Fast pass: `pytest -n auto -m "not slow" -q`
  - Full pass: `pytest -n auto`
- In PRs:
  - If you change expected behavior, include corresponding test edits and clearly describe them.
  - Keep slow tests minimal; prefer unit/integration balance.

## Anti-patterns

- Flaky tests, sleeps for synchronization, reliance on external services, or hidden global state.
