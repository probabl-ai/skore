"""Package that provides APIs to communicate between ``skore`` and ``skore hub``."""

from base64 import b64decode, b64encode
from collections.abc import Iterator
from contextlib import contextmanager
from logging import basicConfig, getLogger

from rich.console import Console
from rich.theme import Theme

__all__ = [
    "Payload",
    "b64_str_to_bytes",
    "bytes_to_b64_str",
    "console",
    "switch_mpl_backend",
]


basicConfig()
logger = getLogger(__name__)

console = Console(
    width=88,
    theme=Theme(
        {
            "repr.str": "cyan",
            "rule.line": "orange1",
            "repr.url": "orange1",
        }
    ),
)


def b64_str_to_bytes(literal: str) -> bytes:
    """Decode the Base64 str object ``literal`` in a bytes."""
    return b64decode(literal.encode("utf-8"))


def bytes_to_b64_str(literal: bytes) -> str:
    """Encode the bytes-like object ``literal`` in a Base64 str."""
    return b64encode(literal).decode("utf-8")


@contextmanager
def switch_mpl_backend(backend: str = "agg") -> Iterator[None]:
    """
    Context-manager for switching ``matplotlib.backend``.

    Notes
    -----
    The ``agg`` backend is a non-interactive backend that can only write to files.
    It is used in ``skore-hub-project`` to generate artifacts where we don't need an
    X display.

    https://matplotlib.org/stable/users/explain/figure/backends.html#selecting-a-backend
    """
    import importlib
    import sys

    import matplotlib
    import matplotlib.pyplot

    original_backend = matplotlib.get_backend()
    original_pyplot_module = sys.modules.pop("matplotlib.pyplot")

    try:
        matplotlib.use(backend)
        importlib.import_module("matplotlib.pyplot")
        yield
    finally:
        sys.modules.pop("matplotlib.pyplot")
        matplotlib.use(original_backend)
        sys.modules["matplotlib.pyplot"] = original_pyplot_module
