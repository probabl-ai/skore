from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from typing import Any

import joblib
import numpy as np

from skore._sklearn.types import _DEFAULT, DataSource


def make_cache_key(
    data_source: DataSource,
    name: str,
    kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any, ...]:
    """Build a cache key.

    Enforce structure (data_source, "predict_time", sanitized_kwargs).
    """
    if data_source not in {"train", "test"}:
        raise ValueError(f"data_source must be 'train' or 'test'; got {data_source!r}")
    if not isinstance(name, str):
        raise TypeError(f"name must be a string; got {type(name)}")
    return (data_source, name, deep_key_sanitize(kwargs))


def deep_key_sanitize(value: Any) -> Any:
    """Recursively normalize a value before inserting it in a cache key tuple.

    This keeps key structures explicit (tuple-based)
    while ensuring values are hashable and deterministic.

    Examples
    --------
    >>> deep_key_sanitize(["mean", "std"])
    ('mean', 'std')

    >>> deep_key_sanitize({"b": [1, 2], "a": {"x": 3}})
    ('mapping', (('a', ('mapping', (('x', 3),))), ('b', (1, 2))))
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
