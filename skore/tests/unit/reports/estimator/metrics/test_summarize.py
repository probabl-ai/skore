import re

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split

from skore import EstimatorReport


@pytest.mark.parametrize("metric", ["public_metric", "_private_metric"])
def test_error_metric_strings(linear_regression_with_test, metric):
    """Check that we raise an error if a metric string is not a valid metric."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    err_msg = re.escape(f"Invalid metric: {metric!r}.")
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.summarize(metric=[metric])


@pytest.mark.parametrize("metric", ["accuracy", "brier_score", "roc_auc", "log_loss"])
def test_binary_classification(forest_binary_classification_with_test, metric):
    """Check the behaviour of the metrics methods available for binary
    classification.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, metric)
    result = getattr(report.metrics, metric)()
    assert isinstance(result, float)
    # check that we hit the cache
    result_with_cache = getattr(report.metrics, metric)()
    assert result == pytest.approx(result_with_cache)

    # check that something was written to the cache
    assert report._cache != {}
    report.clear_cache()

    # check that passing using data outside from the report works and that we they
    # don't come from the cache
    result_external_data = getattr(report.metrics, metric)(
        data_source="X_y", X=X_test, y=y_test
    )
    assert isinstance(result_external_data, float)
    assert result == pytest.approx(result_external_data)
    assert report._cache != {}


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_binary_classification_pr(forest_binary_classification_with_test, metric):
    """Check the behaviour of the precision and recall metrics available for binary
    classification.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, metric)
    result = getattr(report.metrics, metric)()
    assert isinstance(result, dict)
    # check that we hit the cache
    result_with_cache = getattr(report.metrics, metric)()
    assert result == result_with_cache

    # check that something was written to the cache
    assert report._cache != {}
    report.clear_cache()

    # check that passing using data outside from the report works and that we they
    # don't come from the cache
    result_external_data = getattr(report.metrics, metric)(
        data_source="X_y", X=X_test, y=y_test
    )
    assert isinstance(result_external_data, dict)
    assert result == result_external_data
    assert report._cache != {}


@pytest.mark.parametrize("metric", ["r2", "rmse"])
def test_regression(linear_regression_with_test, metric):
    """Check the behaviour of the metrics methods available for regression."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, metric)
    result = getattr(report.metrics, metric)()
    assert isinstance(result, float)
    # check that we hit the cache
    result_with_cache = getattr(report.metrics, metric)()
    assert result == pytest.approx(result_with_cache)

    # check that something was written to the cache
    assert report._cache != {}
    report.clear_cache()

    # check that passing using data outside from the report works and that we they
    # don't come from the cache
    result_external_data = getattr(report.metrics, metric)(
        data_source="X_y", X=X_test, y=y_test
    )
    assert isinstance(result_external_data, float)
    assert result == pytest.approx(result_external_data)
    assert report._cache != {}


def test_data_source_both(forest_binary_classification_data):
    """Check the behaviour of `summarize` with `data_source="both"`."""
    estimator, X, y = forest_binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    result_train = report.metrics.summarize(data_source="train").frame()
    result_test = report.metrics.summarize(data_source="test").frame()
    result_both = report.metrics.summarize(data_source="both").frame()

    assert result_both.columns.tolist() == [
        "RandomForestClassifier (train)",
        "RandomForestClassifier (test)",
    ]
    assert_series_equal(
        result_both["RandomForestClassifier (train)"],
        result_train["RandomForestClassifier"],
        check_names=False,
    )
    assert_series_equal(
        result_both["RandomForestClassifier (test)"],
        result_test["RandomForestClassifier"],
        check_names=False,
    )

    # By default,
    result_both = report.metrics.summarize(
        data_source="both", favorability=True
    ).frame()
    assert result_both.columns.tolist() == [
        "RandomForestClassifier (train)",
        "RandomForestClassifier (test)",
        "Favorability",
    ]


def test_metric_dict(forest_binary_classification_with_test):
    """Test that metric can be passed as a dictionary with custom names."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    # Test with dictionary metric
    metric_dict = {
        "Custom Accuracy": "accuracy",
        "Custom Precision": "precision",
        "Custom R2": get_scorer("neg_mean_absolute_error"),
    }

    result = report.metrics.summarize(metric=metric_dict).frame()

    # Check that custom names are used
    assert "Custom Accuracy" in result.index
    assert "Custom Precision" in result.index
    assert "Custom R2" in result.index

    # Verify the result structure
    assert isinstance(result, pd.DataFrame)
    assert len(result.index) >= 3  # At least our 3 custom metrics


def test_metric_dict_with_callables(linear_regression_with_test):
    """Test that metric dict works with callable functions."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def custom_metric(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    metric_dict = {"R Squared": "r2", "Custom MAE": custom_metric}

    result = report.metrics.summarize(
        metric=metric_dict, metric_kwargs={"response_method": "predict"}
    ).frame()

    # Check that custom names are used
    assert "R Squared" in result.index
    assert "Custom MAE" in result.index

    # Verify the result structure
    assert isinstance(result, pd.DataFrame)
    assert len(result.index) == 2
