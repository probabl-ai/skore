import re

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from skore.sklearn._cross_validation.report import (
    CrossValidationReport,
    _generate_estimator_report,
)
from skore.sklearn._estimator import EstimatorReport


@pytest.fixture
def binary_classification_data():
    """Create a binary classification dataset and return fitted estimator and data."""
    X, y = make_classification(random_state=42)
    return RandomForestClassifier(n_estimators=2, random_state=42), X, y


@pytest.fixture
def binary_classification_data_svc():
    """Create a binary classification dataset and return fitted estimator and data.
    The estimator is a SVC that does not support `predict_proba`.
    """
    X, y = make_classification(random_state=42)
    return SVC(), X, y


@pytest.fixture
def multiclass_classification_data():
    """Create a multiclass classification dataset and return fitted estimator and
    data."""
    X, y = make_classification(
        n_classes=3, n_clusters_per_class=1, random_state=42, n_informative=10
    )
    return RandomForestClassifier(n_estimators=2, random_state=42), X, y


@pytest.fixture
def multiclass_classification_data_svc():
    """Create a multiclass classification dataset and return fitted estimator and
    data. The estimator is a SVC that does not support `predict_proba`.
    """
    X, y = make_classification(
        n_classes=3, n_clusters_per_class=1, random_state=42, n_informative=10
    )
    return SVC(), X, y


@pytest.fixture
def binary_classification_data_pipeline():
    """Create a binary classification dataset and return fitted pipeline and data."""
    X, y = make_classification(random_state=42)
    estimator = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    return estimator, X, y


@pytest.fixture
def regression_data():
    """Create a regression dataset and return fitted estimator and data."""
    X, y = make_regression(random_state=42)
    return LinearRegression(), X, y


@pytest.fixture
def regression_multioutput_data():
    """Create a regression dataset and return fitted estimator and data."""
    X, y = make_regression(n_targets=2, random_state=42)
    return LinearRegression(), X, y


def test_generate_estimator_report(binary_classification_data):
    """Test the behaviour of `_generate_estimator_report`."""
    estimator, X, y = binary_classification_data
    # clone the estimator to avoid a potential side effect even though we check that
    # the report is not altering the estimator
    estimator = clone(estimator)
    train_indices = np.arange(len(X) // 2)
    test_indices = np.arange(len(X) // 2, len(X))
    report = _generate_estimator_report(
        estimator=RandomForestClassifier(n_estimators=2, random_state=42),
        X=X,
        y=y,
        train_indices=train_indices,
        test_indices=test_indices,
    )

    assert isinstance(report, EstimatorReport)
    assert report.estimator is not estimator
    assert isinstance(report.estimator, RandomForestClassifier)
    try:
        check_is_fitted(report.estimator)
    except NotFittedError as exc:
        raise AssertionError("The estimator in the report should be fitted.") from exc
    np.testing.assert_allclose(report.X_train, X[train_indices])
    np.testing.assert_allclose(report.y_train, y[train_indices])
    np.testing.assert_allclose(report.X_test, X[test_indices])
    np.testing.assert_allclose(report.y_test, y[test_indices])


########################################################################################
# Check the general behaviour of the report
########################################################################################


@pytest.mark.parametrize("cv", [5, 10])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_cross_validation_report_attributes(binary_classification_data, cv, n_jobs):
    """Test the attributes of the cross-validation report."""
    estimator, X, y = binary_classification_data
    report = CrossValidationReport(estimator, X, y, cv=cv, n_jobs=n_jobs)
    assert isinstance(report, CrossValidationReport)
    assert isinstance(report.estimator_reports, list)
    for estimator_report in report.estimator_reports:
        assert isinstance(estimator_report, EstimatorReport)
    assert report.X is X
    assert report.y is y
    assert report.n_jobs == n_jobs
    assert len(report.estimator_reports) == cv
    assert report.estimator_name == "RandomForestClassifier"

    err_msg = "attribute is immutable"
    with pytest.raises(AttributeError, match=err_msg):
        report.estimator = LinearRegression()
    with pytest.raises(AttributeError, match=err_msg):
        report.X = X
    with pytest.raises(AttributeError, match=err_msg):
        report.y = y


def test_cross_validation_report_help(capsys, binary_classification_data):
    """Check that the help method writes to the console."""
    estimator, X, y = binary_classification_data
    report = CrossValidationReport(estimator, X, y)

    report.help()
    captured = capsys.readouterr()
    assert f"Tools to diagnose estimator {estimator.__class__.__name__}" in captured.out


def test_cross_validation_report_repr(binary_classification_data):
    """Check that __repr__ returns a string starting with the expected prefix."""
    estimator, X, y = binary_classification_data
    report = CrossValidationReport(estimator, X, y)

    repr_str = repr(report)
    assert "skore.CrossValidationReport" in repr_str
    assert "reporter.help()" in repr_str


@pytest.mark.parametrize(
    "fixture_name, expected_n_keys",
    [
        ("binary_classification_data", 6),
        ("binary_classification_data_svc", 6),
        ("multiclass_classification_data", 8),
        ("regression_data", 2),
    ],
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_estimator_report_cache_predictions(
    request, fixture_name, expected_n_keys, n_jobs
):
    """Check that calling cache_predictions fills the cache."""
    estimator, X, y = request.getfixturevalue(fixture_name)
    report = CrossValidationReport(estimator, X, y, cv=2, n_jobs=n_jobs)
    report.cache_predictions(n_jobs=n_jobs)
    # no effect on the actual cache of the cross-validation report but only on the
    # underlying estimator reports
    assert report._cache == {}

    for estimator_report in report.estimator_reports:
        assert len(estimator_report._cache) == expected_n_keys

    report.clear_cache()
    assert report._cache == {}
    for estimator_report in report.estimator_reports:
        assert estimator_report._cache == {}


########################################################################################
# Check the plot methods
########################################################################################


def test_cross_validation_report_plot_help(capsys, binary_classification_data):
    """Check that the help method writes to the console."""
    estimator, X, y = binary_classification_data
    report = CrossValidationReport(estimator, X, y, cv=2)

    report.metrics.plot.help()
    captured = capsys.readouterr()
    assert "Available plot methods" in captured.out


def test_cross_validation_report_plot_repr(binary_classification_data):
    """Check that __repr__ returns a string starting with the expected prefix."""
    estimator, X, y = binary_classification_data
    report = CrossValidationReport(estimator, X, y, cv=2)

    repr_str = repr(report.metrics.plot)
    assert "skore.CrossValidationReport.metrics.plot" in repr_str
    assert "reporter.metrics.plot.help()" in repr_str


########################################################################################
# Check the metrics methods
########################################################################################


def test_estimator_report_metrics_help(capsys, binary_classification_data):
    """Check that the help method writes to the console."""
    estimator, X, y = binary_classification_data
    report = CrossValidationReport(estimator, X, y, cv=2)

    report.metrics.help()
    captured = capsys.readouterr()
    assert "Available metrics methods" in captured.out


def test_estimator_report_metrics_repr(binary_classification_data):
    """Check that __repr__ returns a string starting with the expected prefix."""
    estimator, X, y = binary_classification_data
    report = CrossValidationReport(estimator, X, y, cv=2)

    repr_str = repr(report.metrics)
    assert "skore.CrossValidationReport.metrics" in repr_str
    assert "reporter.metrics.help()" in repr_str


def _normalize_metric_name(column):
    """Helper to normalize the metric name present in a pandas column that could be
    a multi-index or single-index."""
    # if we have a multi-index, then the metric name is on level 0
    s = column[0] if isinstance(column, tuple) else column
    # Remove spaces and underscores
    return re.sub(r"[^a-zA-Z]", "", s.lower())


def _check_results_single_metric(report, metric, expected_n_splits, expected_nb_stats):
    assert hasattr(report.metrics, metric)
    result = getattr(report.metrics, metric)()
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == expected_n_splits
    # check that we hit the cache
    result_with_cache = getattr(report.metrics, metric)()
    pd.testing.assert_frame_equal(result, result_with_cache)

    # check that the index contains the expected split names
    split_names = result.index.get_level_values(1).unique()
    expected_split_names = [f"Split #{i}" for i in range(expected_n_splits)]
    assert list(split_names) == expected_split_names

    # check that something was written to the cache
    assert report._cache != {}
    report.clear_cache()

    _check_results_report_metrics(result, [metric], expected_nb_stats)

    # check the aggregate parameter
    stats = ["mean", "std"]
    result = getattr(report.metrics, metric)(aggregate=stats)
    # check that the index contains the expected split names
    split_names = result.index.get_level_values(1).unique()
    assert list(split_names) == stats


def _check_results_report_metrics(result, expected_metrics, expected_nb_stats):
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == expected_nb_stats

    normalized_expected = {
        _normalize_metric_name(metric) for metric in expected_metrics
    }
    for column in result.columns:
        normalized_column = _normalize_metric_name(column)
        matches = [
            metric for metric in normalized_expected if metric == normalized_column
        ]
        assert len(matches) == 1, (
            f"No match found for column '{column}' in expected metrics: "
            f" {expected_metrics}"
        )


@pytest.mark.parametrize(
    "metric, nb_stats",
    [
        ("accuracy", 1),
        ("precision", 2),
        ("recall", 2),
        ("brier_score", 1),
        ("roc_auc", 1),
        ("log_loss", 1),
    ],
)
def test_cross_validation_report_metrics_binary_classification(
    binary_classification_data, metric, nb_stats
):
    """Check the behaviour of the metrics methods available for binary
    classification.
    """
    (estimator, X, y), cv = binary_classification_data, 2
    report = CrossValidationReport(estimator, X, y, cv=cv)
    _check_results_single_metric(report, metric, cv, nb_stats)


@pytest.mark.parametrize(
    "metric, nb_stats",
    [
        ("accuracy", 1),
        ("precision", 3),
        ("recall", 3),
        ("roc_auc", 3),
        ("log_loss", 1),
    ],
)
def test_cross_validation_report_metrics_multiclass_classification(
    multiclass_classification_data, metric, nb_stats
):
    """Check the behaviour of the metrics methods available for multiclass
    classification.
    """
    (estimator, X, y), cv = multiclass_classification_data, 2
    report = CrossValidationReport(estimator, X, y, cv=cv)
    _check_results_single_metric(report, metric, cv, nb_stats)


@pytest.mark.parametrize("metric, nb_stats", [("r2", 1), ("rmse", 1)])
def test_cross_validation_report_metrics_regression(regression_data, metric, nb_stats):
    """Check the behaviour of the metrics methods available for regression."""
    (estimator, X, y), cv = regression_data, 2
    report = CrossValidationReport(estimator, X, y, cv=cv)
    _check_results_single_metric(report, metric, cv, nb_stats)


@pytest.mark.parametrize("metric, nb_stats", [("r2", 2), ("rmse", 2)])
def test_cross_validation_report_metrics_regression_multioutput(
    regression_multioutput_data, metric, nb_stats
):
    """Check the behaviour of the metrics methods available for regression."""
    (estimator, X, y), cv = regression_multioutput_data, 2
    report = CrossValidationReport(estimator, X, y, cv=cv)
    _check_results_single_metric(report, metric, cv, nb_stats)
