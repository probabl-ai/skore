from io import BytesIO
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from skore import ComparisonReport, EstimatorReport


@pytest.fixture
def binary_classification_model():
    """Create a binary classification dataset and return fitted estimator and data."""
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return LogisticRegression(random_state=42), X_train, X_test, y_train, y_test


@pytest.fixture
def regression_model():
    """Create a binary classification dataset and return fitted estimator and data."""
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return LinearRegression(), X_train, X_test, y_train, y_test


def test_comparison_report_init_wrong_parameters(binary_classification_model):
    """If the input is not valid, raise."""

    estimator, _, X_test, _, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )

    with pytest.raises(TypeError, match="Expected reports to be an iterable"):
        ComparisonReport(estimator_report)

    with pytest.raises(
        ValueError, match="At least 2 instances of EstimatorReport are needed"
    ):
        ComparisonReport([estimator_report])

    with pytest.raises(
        TypeError, match="Only instances of EstimatorReport are allowed"
    ):
        ComparisonReport([None, estimator_report])


def test_comparison_report_init_deepcopy(binary_classification_model):
    """If an estimator report is modified outside of the comparator, it is not modified
    inside the comparator."""
    estimator, _, X_test, _, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )

    comp = ComparisonReport([estimator_report, estimator_report])

    # check if the deepcopy work well
    assert comp.estimator_reports_[0]._hash == estimator_report._hash

    # modify the estimator report outside of the comparator
    estimator_report._hash = 0

    # check if there is no impact on the estimator report in the comparator (its value
    # has not changed)
    assert comp.estimator_reports_[0]._hash != 0


def test_comparison_report_init_without_testing_data(binary_classification_model):
    """Raise a warning if there is no test data (`None`) for any estimator
    report."""
    estimator, _, _, _, _ = binary_classification_model
    estimator_report = EstimatorReport(estimator, fit=False)

    with pytest.raises(ValueError, match="Cannot compare reports without testing data"):
        ComparisonReport([estimator_report, estimator_report])


def test_comparison_report_init_different_ml_usecases(
    binary_classification_model, regression_model
):
    linear_regression_estimator, _, X_test, _, y_test = regression_model
    linear_regression_report = EstimatorReport(
        linear_regression_estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )

    logistic_regression_estimator, _, X_test, _, y_test = binary_classification_model
    logistic_regression_report = EstimatorReport(
        logistic_regression_estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )

    with pytest.raises(
        ValueError, match="Not all estimators are in the same ML usecase"
    ):
        ComparisonReport([linear_regression_report, logistic_regression_report])


def test_comparison_report_init_different_test_data(binary_classification_model):
    estimator, _, X_test, _, y_test = binary_classification_model

    with pytest.raises(
        ValueError, match="Not all estimators have the same testing data"
    ):
        ComparisonReport(
            [
                EstimatorReport(
                    estimator,
                    fit=False,
                    X_test=X_test,
                    y_test=y_test,
                ),
                EstimatorReport(
                    estimator,
                    fit=False,
                    X_test=X_test[:-1],
                    y_test=y_test[:-1],
                ),
            ]
        )


def test_comparison_report_init_with_report_names(binary_classification_model):
    estimator, X_train, X_test, y_train, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    comp = ComparisonReport(
        [estimator_report, estimator_report], report_names=["r1", "r2"]
    )

    pd.testing.assert_index_equal(
        comp.metrics.accuracy().index,
        pd.MultiIndex.from_tuples(enumerate(["r1", "r2"]), names=[None, "Estimator"]),
    )


def test_comparison_report_init_without_report_names(binary_classification_model):
    estimator, X_train, X_test, y_train, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    comp = ComparisonReport([estimator_report, estimator_report])

    pd.testing.assert_index_equal(
        comp.metrics.accuracy().index,
        pd.MultiIndex.from_tuples(
            enumerate(["LogisticRegression", "LogisticRegression"]),
            names=[None, "Estimator"],
        ),
    )


def test_comparison_report_init_with_invalid_report_names(binary_classification_model):
    estimator, _, X_test, _, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )

    with pytest.raises(
        ValueError, match="There should be as many report names as there are reports"
    ):
        ComparisonReport([estimator_report, estimator_report], report_names=["r1"])


def test_comparison_report_help(capsys, binary_classification_model):
    estimator, _, X_test, _, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )

    ComparisonReport([estimator_report, estimator_report]).help()

    assert "Tools to compare estimators" in capsys.readouterr().out


def test_comparison_report_repr(binary_classification_model):
    estimator, _, X_test, _, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )

    repr_str = repr(ComparisonReport([estimator_report, estimator_report]))

    assert "skore.ComparisonReport" in repr_str
    assert "help()" in repr_str


def test_comparison_report_pickle(tmp_path, binary_classification_model):
    """Check that we can pickle a comparison report."""
    estimator, _, X_test, _, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        fit=False,
        X_test=X_test,
        y_test=y_test,
    )

    with BytesIO() as stream:
        joblib.dump(ComparisonReport([estimator_report, estimator_report]), stream)


@pytest.mark.parametrize(
    "metric_name,metric_value",
    [
        ("accuracy", 1),
        ("precision", 2),
        ("recall", 3),
        ("brier_score", 4),
        ("roc_auc", 5),
        ("log_loss", 6),
    ],
)
def test_estimator_report_metrics_binary_classification(
    monkeypatch, metric_name, metric_value, binary_classification_model
):
    def dummy(*args, **kwargs):
        return pd.DataFrame([[metric_value]])

    monkeypatch.setattr(
        f"skore.sklearn._estimator.metrics_accessor._MetricsAccessor.{metric_name}",
        dummy,
    )

    estimator, X_train, X_test, y_train, y_test = binary_classification_model
    estimator_report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    comp = ComparisonReport([estimator_report, estimator_report])

    # ensure metric is valid
    result = getattr(comp.metrics, metric_name)()
    np.testing.assert_array_equal(result.to_numpy(), [[metric_value], [metric_value]])

    # ensure metric is valid even from the cache
    result = getattr(comp.metrics, metric_name)()
    np.testing.assert_array_equal(result.to_numpy(), [[metric_value], [metric_value]])


@pytest.mark.parametrize(
    "metric_name,metric_value",
    [
        ("rmse", 1),
        ("r2", 2),
    ],
)
def test_estimator_report_metrics_linear_regression(
    monkeypatch, metric_name, metric_value, regression_model
):
    def dummy(*args, **kwargs):
        return pd.DataFrame([[metric_value]])

    monkeypatch.setattr(
        f"skore.sklearn._estimator.metrics_accessor._MetricsAccessor.{metric_name}",
        dummy,
    )

    estimator, X_train, X_test, y_train, y_test = regression_model
    estimator_report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    comp = ComparisonReport([estimator_report, estimator_report])

    # ensure metric is valid
    result = getattr(comp.metrics, metric_name)()
    np.testing.assert_array_equal(result.to_numpy(), [[metric_value], [metric_value]])

    # ensure metric is valid even from the cache
    result = getattr(comp.metrics, metric_name)()
    np.testing.assert_array_equal(result.to_numpy(), [[metric_value], [metric_value]])
