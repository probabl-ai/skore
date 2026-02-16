from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from typing import Any

import joblib
import numpy as np

from skore._sklearn.types import _DEFAULT


def deep_key_sanitize(value: Any) -> Any:
    """Recursively normalize a value before inserting it in a cache key tuple.

    This keeps key structures explicit (tuple-based)
    while ensuring values are hashable and deterministic.
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
        return ("dtype", str(value))
    if isinstance(value, np.ndarray):
        return ("array", joblib.hash(value))
    if callable(value):
        return ("callable", joblib.hash(value))
    if isinstance(value, Mapping):
        all_keys_str = all(isinstance(key, str) for key in value)
        if not all_keys_str:
            raise TypeError("Only string keys are support for mappings")
        kv_pairs = ((key, deep_key_sanitize(item)) for key, item in value.items())
        return ("mapping", tuple(sorted(kv_pairs, key=lambda kv: kv[0])))
    if isinstance(value, Set):
        items = (deep_key_sanitize(item) for item in value)
        return ("set", tuple(sorted(items, key=hash)))
    if isinstance(value, Sequence) or hasattr(value, "__iter__"):
        return tuple(deep_key_sanitize(item) for item in value)

    raise TypeError(f"Unsupported type: {type(value)}")
