"""ml_core public API.

This package exposes a minimal, stable surface intended for use by
external code. The following symbols are part of the supported API:

- Models: ``BaseLitModule``
- Compositions: ``CriterionsComposition``, ``MetricsComposition``
- Utilities: ``instantiate_callbacks``, ``instantiate_loggers``,
  ``extras``, ``task_wrapper``, ``get_metric_value``, ``RankedLogger``

Stability policy: these symbols are considered stable for minor releases.
Additions may occur; removals or breaking changes will be communicated in
release notes.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    # Models
    "BaseLitModule",
    # Compositions
    "CriterionsComposition",
    "MetricsComposition",
    # Utilities
    "instantiate_callbacks",
    "instantiate_loggers",
    "extras",
    "task_wrapper",
    "get_metric_value",
    "RankedLogger",
]


def __getattr__(
    name: str,
) -> Any:  # PEP 562 lazy imports to avoid heavy deps at import time
    """Lazily resolve public API attributes when they are first accessed."""
    if name == "BaseLitModule":
        from ml_core.models.base_module import BaseLitModule

        return BaseLitModule

    if name == "CriterionsComposition":
        from ml_core.models.utils import CriterionsComposition

        return CriterionsComposition

    if name == "MetricsComposition":
        from ml_core.models.utils import MetricsComposition

        return MetricsComposition

    if name == "instantiate_callbacks":
        from ml_core.utils.instantiators import instantiate_callbacks

        return instantiate_callbacks

    if name == "instantiate_loggers":
        from ml_core.utils.instantiators import instantiate_loggers

        return instantiate_loggers

    if name == "extras":
        from ml_core.utils.utils import extras

        return extras

    if name == "task_wrapper":
        from ml_core.utils.utils import task_wrapper

        return task_wrapper

    if name == "get_metric_value":
        from ml_core.utils.utils import get_metric_value

        return get_metric_value

    if name == "RankedLogger":
        from ml_core.utils.pylogger import RankedLogger

        return RankedLogger

    raise AttributeError(f"module 'ml_core' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return sorted list of names for interactive discovery and tab completion."""
    return sorted(list(globals().keys()) + __all__)
