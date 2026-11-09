import joblib
import numpy as np
import pytest

from skore._utils.cache_key import deep_key_sanitize, make_cache_key


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


def test_make_cache_key_rejects_invalid_parts():
    """Cache keys validate scope, data source, and metric name."""
    key = make_cache_key("metrics", "test", "r2", {"average": "macro"})
    assert key[:3] == ("metrics", "test", "r2")

    with pytest.raises(ValueError, match="scope must be one of"):
        make_cache_key("other", "test", "r2")
    with pytest.raises(ValueError, match="data_source must be"):
        make_cache_key("metrics", "both", "r2")
    with pytest.raises(TypeError, match="name must be a string"):
        make_cache_key("metrics", "test", None)


def _score():
    return 1


def test_deep_key_sanitize_dtype_callable_and_iterator():
    """NumPy dtypes, callables, and iterators become deterministic key parts."""
    assert deep_key_sanitize(np.dtype("float64")) == ("dtype", "float64")
    assert deep_key_sanitize(_score) == ("callable", joblib.hash(_score))
    assert deep_key_sanitize(iter([1, 2])) == (1, 2)


def test_deep_key_sanitize_rejects_unsupported_values():
    """Mappings need string keys, and arbitrary objects cannot be keys."""
    with pytest.raises(TypeError, match="Only string keys"):
        deep_key_sanitize({1: "a"})
    with pytest.raises(TypeError, match="Unsupported type"):
        deep_key_sanitize(object())
