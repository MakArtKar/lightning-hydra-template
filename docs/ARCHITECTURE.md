## Architecture Overview (C4‑lite)

This repository provides a structured, extensible ML core built around PyTorch Lightning and Hydra. The goal is to keep clear boundaries between contexts, make extension points explicit, and enable enforcement of import rules via import‑linter.

### System context

- Consumers interact through CLI entrypoints (`ml_core/train.py`, `ml_core/eval.py`) configured by Hydra (`configs/`).
- The core library (`ml_core/`) provides reusable data, model, transform, and utility building blocks.
- Experiments are configured in YAML and executed via Lightning `Trainer` with optional loggers/callbacks.

### Container view

- `ml_core/` — the main library
  - `ml_core.data` — data modules and dataloaders
  - `ml_core.models` — Lightning modules, model components, losses/metrics composition
  - `ml_core.transforms` — composable dict‑based transforms (image and generic)
  - `ml_core.utils` — logging, instantiation helpers, misc utilities
- `configs/` — Hydra configs (trainer, callbacks, data, model, logger, debug, paths)
- `tests/` — fast tests validating configuration loading and core behaviors

### Component view and extension points

- `BaseDataModule` (`ml_core.data.base_datamodule.BaseDataModule`)
  - Wraps dataset preparation/splitting and builds `DataLoader`s
  - Extension: subclass for new datasets or inject a `transform` callable
- `BaseLitModule` (`ml_core.models.base_module.BaseLitModule`)
  - Wires a `forward_fn`, losses (`CriterionsComposition`), optional metrics, and optim/scheduler factories
  - Extension: provide your forward function and compositions; override Lightning hooks sparingly
- Transforms (`ml_core.transforms.*`)
  - Dict‑to‑dict batch transforms (e.g., `RenameTransform`, `ComposeTransform`, `WrapTransform`, `TorchVisionTransform`)
  - Extension: compose or add new transforms for preprocessing/feature construction
- Hydra configuration
  - Instantiate components via config; keep code decoupled from experiment setup

### Boundaries and allowed dependencies

The library is organized into four contexts with strict dependency flow. Allowed imports:

- `ml_core.transforms` → `ml_core.utils`
- `ml_core.data` → `ml_core.transforms`, `ml_core.utils`
- `ml_core.models` → `ml_core.data`, `ml_core.transforms`, `ml_core.utils`
- `ml_core.utils` → (no dependencies on other `ml_core.*` contexts)

Notes:

- No cyclic or reverse dependencies (e.g., `utils` must not import from `data/models/transforms`).
- Tests may import across boundaries for verification only.

These boundaries are intended to be enforced with import‑linter contracts (see Task 10). Example contracts to be configured later:

```ini
[importlinter:contract:ml_core_layers]
name = Enforce layered architecture in ml_core
type = layers
layers =
    ml_core.utils
    ml_core.transforms
    ml_core.data
    ml_core.models
```

### Public API surface

The stable, documented import surface will live in `ml_core/__init__.py` (see Task 15). Internal modules may evolve; consumers should import only from the public API unless developing inside this repo.

### Configuration and execution

- Use Hydra configs in `configs/` to select data modules, models, transforms, callbacks, trainers, and loggers.
- Entry points:
  - `python ml_core/train.py` for training
  - `python ml_core/eval.py` for evaluation
- Debug presets in `configs/debug/` (e.g., `debug=fdr`, `debug=limit`) enable quick local runs.

### Non‑goals

- This document is implementation‑agnostic; it does not prescribe specific architectures beyond boundaries and extension points.
- Actual import‑linter configuration and public API wiring are addressed in Task 10 and Task 15 respectively.

### Verification

The following should hold:

- Boundaries above are explicitly stated here and later enforced by import‑linter.
- Public API is referenced (and will be wired in Task 15).
