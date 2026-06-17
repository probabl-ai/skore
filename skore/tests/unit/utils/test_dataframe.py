import numpy as np
import pandas as pd
import polars as pl
import pytest
import scipy.sparse as sp

from skore._utils._dataframe import (
    _concat_vertical_frames,
    _normalize_X_as_dataframe,
    _normalize_y_as_dataframe,
)


def test_concat_vertical_frames_resets_pandas_index():
    X_train = pd.DataFrame({"a": [1, 2]}, index=[10, 11])
    X_test = pd.DataFrame({"a": [3, 4]}, index=[10, 11])
    result = _concat_vertical_frames(X_train, X_test)
    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == [0, 1, 2, 3]


def test_concat_vertical_frames_polars():
    X_train = pl.DataFrame({"a": [1, 2]})
    X_test = pl.DataFrame({"a": [3, 4]})
    result = _concat_vertical_frames(X_train, X_test)
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (4, 1)


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
