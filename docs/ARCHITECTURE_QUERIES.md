## Architecture queries checklist

Concise prompts to quickly understand and validate the system design. Use these questions during reviews, refactors, or when adding new features. Link to exact files and focused code ranges when answering.

### Core invariants

- **What must always hold?** Examples: device placement, seed control, deterministic data splits, idempotent evaluation, reproducible configs.
- **Where are they enforced?** Point to assertions, guards, and tests (e.g., `ml_core/models/base_module.py`, `ml_core/data/base_datamodule.py`).
- **How are violations detected?** Unit tests, type checks, runtime asserts, or callbacks.
- Quick scans:
  - `rg -n "assert|raise|invariant|validate" /Users/Artem.Makoian/work/bayes_group/lightning-hydra-template/ml_core`

### Module dependencies

- **What layers may depend on which?** Define allowed directions (e.g., `utils` -> used by all; `models` must not import `data`; configs should not import code).
- **Where are rules documented/enforced?** Link to docs or tooling (e.g., import-linter, mypy boundaries).
- Quick scans:
  - `rg -n "^from ml_core|^import ml_core" /Users/Artem.Makoian/work/bayes_group/lightning-hydra-template`
  - See `local/ai_coding_init/10_import_linter_config.md` and `local/ai_coding_init/11_mypy_config.md`.

### Extension points and stability

- **Where do users extend?**
  - Lightning module base: `ml_core/models/base_module.py:BaseLitModule`
  - Data modules: `ml_core/data/base_datamodule.py`
  - Transforms: `ml_core/transforms/base.py`, `ml_core/transforms/image.py`
  - Config groups: `configs/model/*`, `configs/data/*`, `configs/trainer/*`, `configs/callbacks/*`, `configs/logger/*`
- **Stability guarantees:** Which hooks/APIs are stable vs. internal? What changes require a spec or ADR?
- **Contract for new models/data:** Required hooks, expected return types, config keys.

### Performance and constraints

- **Targets and budgets:** Throughput (steps/s), max epoch time, memory footprint, dataloader utilization, artifact sizes.
- **Bottleneck diagnostics:** Profilers, callbacks, or logs to check (see `configs/debug/profiler.yaml`, `configs/callbacks/*`).
- **Repro steps:** Command and flags used to measure. Prefer `.venv` and absolute paths.

### Data contracts

- **Inputs/outputs:** Shapes, dtypes, ranges; batch structure; masking; device expectations.
- **Schemas:** Config keys in `configs/*`, required/optional fields, defaults.
- **Failure modes:** Missing data, NaNs/Infs, empty batches, incompatible shapes; where handled.
- Quick scans:
  - `rg -n "->\s*torch\.Tensor|torch\.Tensor\]" /Users/Artem.Makoian/work/bayes_group/lightning-hydra-template/ml_core --type py`
  - `rg -n "batch\[|x\]|y\]" /Users/Artem.Makoian/work/bayes_group/lightning-hydra-template/ml_core --type py`

### Hydra configuration boundaries

- **Config groups:** `configs/train.yaml`, `configs/model/*.yaml`, `configs/data/*.yaml`, `configs/trainer/*.yaml`, `configs/callbacks/*.yaml`, `configs/logger/*.yaml`.
- **Overridable parameters:** Which are safe to override at CLI? Which require code changes?
- **Composition rules:** Defaults list, required groups, and key merge behavior.

### References

- `docs/ARCHITECTURE.md` — system overview and module map
- `docs/CONTEXT.md` — context citation rules and reproducible commands
- `docs/TESTING.md` — testing strategy and local/CI commands
- `AGENTS.md` — pre-flight questions and execution loop
- `docs/SPEC_TEMPLATE.md` — write specs for non-trivial changes
