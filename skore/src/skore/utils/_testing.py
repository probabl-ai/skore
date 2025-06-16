import contextlib
import copy
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.legend import Legend
from sklearn.base import BaseEstimator, ClassifierMixin

from skore.sklearn._plot.metrics.precision_recall_curve import (
    PrecisionRecallCurveDisplay,
)
from skore.sklearn._plot.metrics.roc_curve import RocCurveDisplay


@contextlib.contextmanager
def check_cache_changed(value):
    """Assert that `value` has changed during context execution."""
    initial_value = copy.copy(value)
    yield
    assert value != initial_value


@contextlib.contextmanager
def check_cache_unchanged(value):
    """Assert that `value` has not changed during context execution."""
    initial_value = copy.copy(value)
    yield
    assert value == initial_value


class MockEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self, *, error, n_call=0, fail_after_n_clone=3):
        self.error = error
        self.n_call = n_call
        self.fail_after_n_clone = fail_after_n_clone

    def fit(self, X, y):
        if self.n_call > self.fail_after_n_clone:
            raise self.error
        self.classes_ = np.unique(y)
        return self

    def __sklearn_clone__(self):
        self.n_call += 1
        return self

    def predict(self, X):
        return np.ones(X.shape[0])


def check_roc_curve_display_data(display: RocCurveDisplay):
    """Check the structure of the display's internal data."""
    assert list(display.roc_curve.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "threshold",
        "fpr",
        "tpr",
    ]
    assert list(display.roc_auc.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "roc_auc",
    ]


def check_precision_recall_curve_display_data(display: PrecisionRecallCurveDisplay):
    """Check the structure of the display's internal data."""
    assert list(display.precision_recall.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "threshold",
        "precision",
        "recall",
    ]
    assert list(display.average_precision.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "average_precision",
    ]


def check_legend_position(ax, *, loc: str, position: Literal["inside", "outside"]):
    """Check the position of the legend in the axes."""
    legend = ax.get_legend()
    assert legend._loc == Legend.codes[loc]
    bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
    if position == "inside":
        assert 0 <= bbox.x0 <= 1
    else:
        assert bbox.x0 >= 1


def check_roc_frame(df, expected_index, expected_data_columns):
    """Check the structure of the ROC curve DataFrame."""
    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns.tolist()) == sorted(expected_index + expected_data_columns)

    for col in df.columns:
        if col in expected_index:
            assert df[col].dtype.name == "category"
        else:
            assert col in expected_data_columns
            assert df[col].dtype == np.float64


def check_precision_recall_frame(df, expected_index, expected_data_columns):
    """Check the structure of the precision-recall curve DataFrame."""
    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns.tolist()) == sorted(expected_index + expected_data_columns)

    for col in df.columns:
        if col in expected_index:
            assert df[col].dtype.name == "category"
        else:
            assert col in expected_data_columns
            assert df[col].dtype == np.float64


def check_prediction_error_frame(df, expected_index, expected_data_columns):
    """Check the structure of a prediction error DataFrame.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to check.
    expected_index : list of str
        The expected index columns.
    expected_data_columns : list of str
        The expected data columns.
    """
    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns.tolist()) == sorted(expected_index + expected_data_columns)

    for col in df.columns:
        if col in expected_index:
            assert df[col].dtype.name == "category"
        else:
            assert col in expected_data_columns
            assert df[col].dtype == np.float64
