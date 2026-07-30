import numpy as np
import pandas as pd
import polars as pl
import pytest
import scipy.sparse as sp

from skore._utils._dataframe import (
    _concat_vertical,
    _normalize_X_as_dataframe,
    _normalize_y_as_dataframe,
)


def test_normalize_X_polars_dataframe():
    X = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    result = _normalize_X_as_dataframe(X)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["a", "b"]


def test_normalize_X_pandas_renames_non_string_columns():
    X = pd.DataFrame({0: [1.0], 1: [2.0]})
    result = _normalize_X_as_dataframe(X)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["0", "1"]


def test_normalize_X_preserves_object_dtype():
    X = np.array([[1, 0, 1], [0, 1, 0]], dtype=object)
    result = _normalize_X_as_dataframe(X)
    assert all(dtype == np.dtype("O") for dtype in result.dtypes)
    assert result.iloc[0, 0] == 1


def test_normalize_X_preserves_int32_dtype():
    X = np.array([[1, 2], [3, 4]], dtype=np.int32)
    result = _normalize_X_as_dataframe(X)
    assert all(dtype == np.dtype("int32") for dtype in result.dtypes)


def test_normalize_X_preserves_bool_dtype():
    X = np.array([[True, False], [False, True]])
    result = _normalize_X_as_dataframe(X)
    assert all(dtype == np.dtype("bool") for dtype in result.dtypes)


def test_normalize_y_polars_unnamed_series():
    y = pl.Series([1.0, 2.0, 3.0])
    result = _normalize_y_as_dataframe(y)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["Target"]


def test_normalize_y_polars_series():
    y = pl.Series("target", [1.0, 2.0, 3.0])
    result = _normalize_y_as_dataframe(y)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["target"]


def test_normalize_y_polars_dataframe():
    y = pl.DataFrame({"Target": [1.0, 2.0]})
    result = _normalize_y_as_dataframe(y)
    assert result is y


def test_normalize_X_sparse_raises():
    X_sparse = sp.csr_matrix(np.random.rand(10, 2))
    with pytest.raises(NotImplementedError, match="not supported for sparse matrices"):
        _normalize_X_as_dataframe(X_sparse)


def test_normalize_y_pandas_series():
    y = pd.Series([1.0, 2.0], name="label")
    result = _normalize_y_as_dataframe(y)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["label"]


def test_concat_vertical_preserves_backend():
    X_pandas = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    X_pandas_concat = _concat_vertical(X_pandas, X_pandas)
    assert isinstance(X_pandas_concat, pd.DataFrame)
    assert X_pandas_concat.shape == (4, 2)

    y_pandas = pd.Series([1, 2], name="Target")
    y_pandas_concat = _concat_vertical(y_pandas, y_pandas)
    assert isinstance(y_pandas_concat, pd.Series)
    assert len(y_pandas_concat) == 4

    X_polars = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    X_polars_concat = _concat_vertical(X_polars, X_polars)
    assert isinstance(X_polars_concat, pl.DataFrame)
    assert X_polars_concat.shape == (4, 2)

    y_polars = pl.Series([1, 2])
    y_polars_concat = _concat_vertical(y_polars, y_polars)
    assert isinstance(y_polars_concat, pl.Series)
    assert len(y_polars_concat) == 4

    X_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_array_concat = _concat_vertical(X_array, X_array)
    assert isinstance(X_array_concat, np.ndarray)
    assert X_array_concat.shape == (4, 2)
