from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from typing import Any

import joblib
import numpy as np

from skore._sklearn.types import _DEFAULT


def deep_key_sanitize(value: Any) -> Any:
    """Recursively normalize a value before inserting it in a cache key tuple.

    Inspired by recursive deep-hash traversal, this keeps key structures explicit
    (tuple-based) while ensuring values are hashable and deterministic.
    """
    if (
        value is None
        or value is _DEFAULT
        or isinstance(value, (str, bytes, int, float, bool))
    ):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.dtype):
        return ("numpy.dtype", str(value))

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return (
            "numpy.ndarray",
            joblib.hash(array),
        )

    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                sorted(
                    (
                        (deep_key_sanitize(key), deep_key_sanitize(item))
                        for key, item in value.items()
                    ),
                    key=lambda item: repr(item[0]),
                )
            ),
        )

    if isinstance(value, Set):
        return (
            "set",
            tuple(sorted((deep_key_sanitize(item) for item in value), key=hash)),
        )

    if isinstance(value, Sequence) or hasattr(value, "__iter__"):
        return tuple(deep_key_sanitize(item) for item in value)

    # raise TypeError(f"Unsupported type: {type(value)}")
    return value
