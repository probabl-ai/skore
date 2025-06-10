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


def check_roc_frame(
    df: pd.DataFrame,
    expected_n_splits: int | None = None,
    multiclass: bool = False,
) -> None:
    """Check the structure of a ROC curve DataFrame.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to check.
    expected_n_splits : int or None, default=None
        The expected number of cross-validation splits.
        If None, does not check the number of splits.
    multiclass : bool, default=False
        Whether the DataFrame is from a multiclass classification.
    """
    assert isinstance(df, pd.DataFrame)

    if not (multiclass):
        expected_columns = [
            "estimator_name",
            "split_index",
            "fpr",
            "tpr",
            "threshold",
            "roc_auc",
        ]
    else:
        expected_columns = [
            "estimator_name",
            "split_index",
            "label",
            "method",
            "fpr",
            "tpr",
            "threshold",
            "roc_auc",
        ]
    assert list(df.columns) == expected_columns

    assert df["estimator_name"].dtype.name == "category"
    if expected_n_splits is not None:
        assert df["split_index"].dtype.name == "category"
    if multiclass:
        assert df["label"].dtype.name == "category"
        assert df["method"].dtype == object
    assert df["fpr"].dtype == np.float64
    assert df["tpr"].dtype == np.float64
    assert df["threshold"].dtype == np.float64
    assert df["roc_auc"].dtype == np.float64

    if expected_n_splits is not None:
        assert df["split_index"].nunique() == expected_n_splits

    if multiclass:
        assert df["method"].unique() == ["OvR"]


def check_precision_recall_frame(
    df: pd.DataFrame,
    expected_n_splits: int | None = None,
    multiclass: bool = False,
) -> None:
    """Check the structure of a precision-recall curve DataFrame.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to check.
    expected_n_splits : int or None, default=None
        The expected number of cross-validation splits.
        If None, does not check the number of splits.
    multiclass : bool, default=False
        Whether the DataFrame is from a multiclass classification.
    """
    assert isinstance(df, pd.DataFrame)

    if not (multiclass):
        expected_columns = [
            "estimator_name",
            "split_index",
            "label",
            "threshold",
            "precision",
            "recall",
            "average_precision",
        ]
    else:
        expected_columns = [
            "estimator_name",
            "split_index",
            "label",
            "method",
            "threshold",
            "precision",
            "recall",
            "average_precision",
        ]
    assert list(df.columns) == expected_columns

    assert df["estimator_name"].dtype.name == "category"
    if expected_n_splits is not None:
        assert df["split_index"].dtype.name == "category"
    if multiclass:
        assert df["label"].dtype.name == "category"
        assert df["method"].dtype == object
    assert df["threshold"].dtype == np.float64
    assert df["precision"].dtype == np.float64
    assert df["recall"].dtype == np.float64
    assert df["average_precision"].dtype == np.float64

    if expected_n_splits is not None:
        assert df["split_index"].nunique() == expected_n_splits

    if multiclass:
        assert df["method"].unique() == ["OvR"]


def check_prediction_error_frame(
    df: pd.DataFrame,
    expected_n_splits: int | None = None,
) -> None:
    """Check the structure of a prediction error DataFrame.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to check.
    expected_n_splits : int or None, default=None
        The expected number of cross-validation splits.
        If None, does not check the number of splits.
    """
    assert isinstance(df, pd.DataFrame)

    expected_columns = [
        "estimator_name",
        "split_index",
        "y_true",
        "y_pred",
        "residuals",
    ]
    assert list(df.columns) == expected_columns

    assert df["estimator_name"].dtype.name == "category"
    if expected_n_splits is not None:
        assert df["split_index"].dtype.name == "category"
    assert df["y_true"].dtype == np.float64
    assert df["y_pred"].dtype == np.float64
    assert df["residuals"].dtype == np.float64

    if expected_n_splits is not None:
        assert df["split_index"].nunique() == expected_n_splits
