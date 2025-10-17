import os
import sys
from typing import List, Optional

from . import __version__


USAGE = """
Usage:
  python -m flash_ml <subcommand> [hydra_overrides...]

Subcommands:
  train    Run training with configs/train.yaml
  eval     Run evaluation with configs/eval.yaml

Examples:
  python -m flash_ml train experiment=examples/mnist_classification trainer.max_epochs=3
  python -m flash_ml train experiment=/abs/path/to/exp.yaml data.num_workers=8 debug=fdr
  python -m flash_ml eval ckpt_path=/abs/path/to/last.ckpt experiment=examples/mnist_classification

Options:
  -h, --help     Show this help and exit
  --version      Show version and exit
""".strip()


def _rewrite_experiment_override(overrides: List[str]) -> List[str]:
    rewritten: List[str] = []
    for item in overrides:
        if item.startswith("experiment="):
            key, value = item.split("=", 1)
            path_value = os.path.expanduser(value)
            is_abs = os.path.isabs(path_value)
            is_yaml = path_value.endswith(".yaml") or path_value.endswith(".yml")
            if is_abs or is_yaml:
                if not value.startswith("@"):
                    rewritten.append(f"{key}=@{value}")
                    continue
        rewritten.append(item)
    return rewritten


def _print_help() -> int:
    print(USAGE)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in {"-h", "--help"}:
        return _print_help()

    if args[0] == "--version":
        print(__version__)
        return 0

    subcommand = args[0]
    overrides = args[1:]

    # Forward all unparsed args to Hydra; only rewrite experiment file paths
    overrides = _rewrite_experiment_override(overrides)

    if subcommand not in {"train", "eval"}:
        print(f"Unknown subcommand: {subcommand}\n")
        return _print_help()

    # Ensure Hydra sees only overrides (without our subcommand) in sys.argv
    sys.argv = [f"flash_ml-{subcommand}"] + overrides

    if subcommand == "train":
        from src.train import main as train_main

        return int(train_main() or 0)

    from src.eval import main as eval_main

    eval_main()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


