import joblib
import numpy as np

from skore._utils._cache_key import deep_key_sanitize


def test_deep_key_sanitize_hashes_numpy_array():
    array = np.array([1, 2, 3])
    sanitized = deep_key_sanitize(array)
    assert sanitized == ("array", joblib.hash(array))


def test_deep_key_sanitize_sequence_returns_tuple_value():
    result = deep_key_sanitize(["mean", "std"])
    assert result == ("mean", "std")


def test_deep_key_sanitize_mapping_is_order_independent():
    left = {"b": np.array([1, 2]), "a": [np.int64(3), {"x": np.float64(1.5)}]}
    right = {"a": [3, {"x": 1.5}], "b": np.array([1, 2])}
    assert deep_key_sanitize(left) == deep_key_sanitize(right)


def test_deep_key_sanitize_set_is_order_independent():
    left = {"foo", "bar", "baz"}
    right = {"baz", "foo", "bar"}
    assert deep_key_sanitize(left) == deep_key_sanitize(right)
