# Architecture Overview

This repository provides a structured, extensible ML core built around PyTorch Lightning and Hydra. The goal is to keep clear boundaries between contexts, make extension points explicit, and enable enforcement of import rules via import-linter.

## System Context

Users interact with the system through:
- **CLI entrypoints**: `ml_core/train.py`, `ml_core/eval.py`
- **Configuration**: Hydra configs in `configs/`
- **Core library**: `ml_core/` provides reusable building blocks

Experiments are configured in YAML and executed via Lightning `Trainer` with optional loggers and callbacks.

## Container View

### Main library: `ml_core/`
- **`ml_core.data`** — data modules and dataloaders
- **`ml_core.models`** — Lightning modules, model components, losses/metrics composition
- **`ml_core.transforms`** — composable dict-based transforms (image and generic)
- **`ml_core.utils`** — logging, instantiation helpers, misc utilities

### Configuration: `configs/`
Hydra configs for trainer, callbacks, data, model, logger, debug profiles, and paths

### Tests: `tests/`
Fast tests validating configuration loading and core behaviors

## Component View and Extension Points

### `BaseDataModule` (`ml_core.data.base_datamodule.BaseDataModule`)
- Wraps dataset preparation/splitting and builds `DataLoader`s
- **Extension**: Subclass for new datasets or inject a `transform` callable

### `BaseLitModule` (`ml_core.models.base_module.BaseLitModule`)
- Wires a `forward_fn`, losses (`CriterionsComposition`), optional metrics, and optimizer/scheduler factories
- **Extension**: Provide your forward function and compositions; override Lightning hooks sparingly

### Transforms (`ml_core.transforms.*`)
- Dict-to-dict batch transforms (e.g., `RenameTransform`, `ComposeTransform`, `WrapTransform`, `TorchVisionTransform`)
- **Extension**: Compose or add new transforms for preprocessing/feature construction

### Hydra Configuration
- Instantiate components via config; keep code decoupled from experiment setup
- Override parameters at CLI for quick experimentation

## Boundaries and Allowed Dependencies

The library is organized into four contexts with strict dependency flow:

```
ml_core.utils         (base layer - no ml_core dependencies)
    ↑
ml_core.transforms    (may use utils)
    ↑
ml_core.data         (may use transforms, utils)
    ↑
ml_core.models       (may use data, transforms, utils)
```

### Allowed imports
- `ml_core.transforms` → `ml_core.utils`
- `ml_core.data` → `ml_core.transforms`, `ml_core.utils`
- `ml_core.models` → `ml_core.data`, `ml_core.transforms`, `ml_core.utils`
- `ml_core.utils` → (no dependencies on other `ml_core.*` contexts)

### Rules
- No cyclic or reverse dependencies (e.g., `utils` must not import from `data/models/transforms`)
- Tests may import across boundaries for verification only
- These boundaries are enforced with import-linter (see `architecture/importlinter.ini`)

## Public API Surface

The stable, documented import surface lives in `ml_core/__init__.py`. Internal modules may evolve; external consumers should import only from the public API unless developing inside this repo.

Example usage:
```python
from ml_core import BaseLitModule, BaseDataModule
```

## Configuration and Execution

### Hydra configs
Use configs in `configs/` to select data modules, models, transforms, callbacks, trainers, and loggers.

### Entry points
```bash
# Training
python ml_core/train.py --config-dir configs

# Evaluation
python ml_core/eval.py --config-dir configs
```

### Debug presets
Available in `configs/debug/`:
- `debug=fdr` — fast dev run (1 batch)
- `debug=limit` — limit data size
- `debug=profiler` — enable profiling
- `debug=overfit` — overfit to 1 batch

## Non-Goals

- This document is implementation-agnostic; it does not prescribe specific model architectures beyond boundaries and extension points
- Implementation details of specific models/datasets should be documented in their respective modules

## Verification

Verify architecture constraints:

```bash
# Check import boundaries
lint-imports

# Run tests
make test

# Check type annotations
mypy ml_core
```
