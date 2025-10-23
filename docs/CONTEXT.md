## Context engineering guide

Keep context lean, precise, and reproducible. Use exact file paths and focused code ranges to minimize cognitive load during reviews and agent runs.

### Essentials

- **Use exact paths and symbols**: Reference files and definitions precisely, e.g., `ml_core/models/base_module.py:BaseLitModule` or `ml_core/train.py:main`.
- **Prefer small ranges (15–40 lines)**: When citing code, show only the smallest meaningful slice instead of whole files or giant diffs.
- **Search with ripgrep patterns (not vague text)**: Formulate specific queries that narrow scope.
- **Provide reproducible commands**: Use `.venv`, absolute repository path, and include all flags/options needed to reproduce results.

### Practical patterns

- **Find class/function definitions**
  - `rg -n "class BaseLitModule" /Users/Artem.Makoian/work/bayes_group/lightning-hydra-template/ml_core`
  - `rg -n "def (train|validation)_step\(" /Users/Artem.Makoian/work/bayes_group/lightning-hydra-template/ml_core --pcre2`
- **Locate Hydra configs and specific keys**
  - `rg -n "trainer:|callbacks:|logger:" /Users/Artem.Makoian/work/bayes_group/lightning-hydra-template/configs`
- **Trace import relationships**
  - `rg -n "^from ml_core|^import ml_core" /Users/Artem.Makoian/work/bayes_group/lightning-hydra-template`

### Reproducible environment and runs

Use absolute paths and `.venv`.

```bash
cd /Users/Artem.Makoian/work/bayes_group/lightning-hydra-template
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Quick functional smoke
python ml_core/train.py --config-dir configs debug=fdr

# Local checks
make format
make test
```

### Code citation guidance

- **Do**: Reference exact locations: `ml_core/models/base_module.py:BaseLitModule` and include a 15–40 line excerpt around the change.
- **Avoid**: Entire files, unbounded logs, or vague descriptions like “trainer file logic.”
- **Cross-link** related docs when helpful rather than duplicating content.

### When to include architecture context

Provide a short note explaining how a change affects invariants, module dependencies, or extension points (see `docs/ARCHITECTURE_QUERIES.md`). Link to `docs/ARCHITECTURE.md` for system overview and to `docs/SPEC_TEMPLATE.md` when proposing non-trivial changes.

### References

- `docs/ARCHITECTURE.md` — high-level structure
- `docs/SPEC_TEMPLATE.md` — write specs for larger changes
- `docs/TESTING.md` — local/CI testing guidance
- `AGENTS.md` — pre-flight questions and execution loop
