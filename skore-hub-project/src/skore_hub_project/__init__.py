"""Package that provides APIs to communicate between ``skore`` and ``skore hub``."""

import logging
from base64 import b64decode, b64encode

from rich.console import Console
from rich.theme import Theme

__all__ = [
    "Payload",
    "b64_str_to_bytes",
    "bytes_to_b64_str",
    "console",
]


logging.basicConfig()
logger = logging.getLogger(__name__)

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
