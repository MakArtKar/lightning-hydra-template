# Testing Policy

**Tests-as-Specifications**: This repository treats tests as executable specifications. A change is correct only if it satisfies the existing test suite or explicitly updates the specification with reviewed test changes.

## Core Principles

### Tests are specifications
- Behavior is defined by tests
- Code changes must pass existing tests
- If behavior changes, update tests explicitly in the same PR and call this out in the description

### Deterministic and hermetic
- Tests must be repeatable across machines and runs
- Seed all randomness where applicable; avoid time- or network-dependent flakiness
- No external network or filesystem dependencies unless mocked/faked
- Prefer `tmp_path` for I/O operations

### Small and focused
- Prefer many small, readable tests over large end-to-end-only suites
- Each test should verify one clear behavior

## Speed and Parallelism

### Parallel execution

Use pytest with xdist to parallelize tests:

```bash
pytest -n auto -q
```

### Marking slow tests

Mark genuinely long-running tests with the `slow` marker and keep them minimal:

```python
import pytest

@pytest.mark.slow
def test_expensive_path():
    """Test that requires significant time/resources."""
    ...
```

### Running tests efficiently

During development, run only fast tests:

```bash
make test             # runs: pytest -n auto -m "not slow" -q
make test-full        # runs: pytest -n auto
```

Or use pytest directly:

```bash
pytest -n auto -m "not slow" -q  # fast tests only
pytest -n auto                    # all tests
```

### Configuration

- `pyproject.toml` enables `--strict-markers` and defines the `slow` marker
- xdist is installed via `requirements-dev.txt`

## Coverage

Aim for meaningful coverage of critical logic; avoid chasing 100% without value.

### Running coverage locally

```bash
pytest -n auto --cov=ml_core --cov-report=term-missing
```

### Interpreting results

- Focus on covering critical logic paths
- 100% coverage is not the goal; meaningful coverage is
- Uncovered lines in error handling may be acceptable

## Fixtures and Patterns

### Built-in fixtures

Prefer pytest's built-in fixtures:
- `tmp_path` — temporary directory for file operations
- `monkeypatch` — modify objects, dictionaries, environment variables
- `capsys` / `caplog` — capture output and logs

### Repository fixtures

Available in `tests/conftest.py`:
- `cfg_train` / `cfg_train_global` — Hydra training configs
- `cfg_eval` / `cfg_eval_global` — Hydra evaluation configs

### Best practices

- Keep fixtures simple, explicit, and reusable
- Avoid hidden side effects
- Document fixture behavior with docstrings

## Development Workflow

### During development

```bash
make test        # Fast pass: pytest -n auto -m "not slow" -q
make test-full   # Full pass: pytest -n auto
```

### Before committing

```bash
make format      # Run pre-commit hooks
make test        # Run fast tests
```

### In pull requests

- If you change expected behavior, include corresponding test edits
- Clearly describe test changes in the PR description
- Keep slow tests minimal; maintain unit/integration balance

## Anti-patterns

Avoid these common pitfalls:

- ❌ Flaky tests that pass/fail randomly
- ❌ Using `sleep()` for synchronization
- ❌ Reliance on external services without mocks
- ❌ Hidden global state that affects test independence
- ❌ Tests that depend on execution order
- ❌ Over-mocking that tests nothing meaningful
