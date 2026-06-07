"""Module that provides APIs to communicate between ``skore`` and ``skore hub``."""

from base64 import b64decode, b64encode
from logging import basicConfig, getLogger

__all__ = ["b64_str_to_bytes", "bytes_to_b64_str"]


basicConfig()
logger = getLogger(__name__)


def b64_str_to_bytes(literal: str) -> bytes:
    """Decode the Base64 str object ``literal`` in a bytes."""
    return b64decode(literal.encode("utf-8"))


def bytes_to_b64_str(literal: bytes) -> str:
    """Encode the bytes-like object ``literal`` in a Base64 str."""
    return b64encode(literal).decode("utf-8")
