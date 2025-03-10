import warnings


warnings.warn(
    "The module `pytensor.graph.opt` is deprecated; use `pytensor.graph.rewriting.basic` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pytensor.graph.rewriting.basic import *  # noqa: F401 E402 F403
from pytensor.graph.rewriting.basic import DEPRECATED_NAMES  # noqa: F401 E402 F403


def __getattr__(name):
    """Intercept module-level attribute access of deprecated symbols.

    Adapted from https://stackoverflow.com/a/55139609/3006474.

    """
    global DEPRECATED_NAMES

    from warnings import warn

    for old_name, msg, old_object in DEPRECATED_NAMES:
        if name == old_name:
            warn(msg, DeprecationWarning, stacklevel=2)
            return old_object

    raise AttributeError(f"module {__name__} has no attribute {name}")
