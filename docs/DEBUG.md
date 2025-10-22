## Debug profile quickstart

This repository includes Hydra debug profiles to enable fast, iterative runs during development.

### Setup a local virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run a fast debug training

Use the fast dev run profile (`fdr`) to quickly validate your setup and code changes:

```bash
python -m ml_core.train experiment=example debug=fdr
```

### Adjust debug settings

All debug profiles live under `configs/debug/`:

- `configs/debug/fdr.yaml` — fast dev run profile (reduced steps/epochs, quick feedback)
- `configs/debug/limit.yaml` — dataset/loader size limits for faster iterations
- `configs/debug/profiler.yaml` — profiling options
- `configs/debug/overfit.yaml` — single-batch overfit for sanity checks

Tune these profiles or compose them in your runs to suit your local workflow.

### Notes

- You can combine profiles as needed via Hydra, e.g., add tags or limits.
- For more context, see the training entrypoint at `ml_core/train.py`.
