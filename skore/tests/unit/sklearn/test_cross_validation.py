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
@pytest.mark.parametrize(
    "fixture_name",
    ["binary_classification_data", "binary_classification_data_pipeline"],
)
def test_cross_validation_report_attributes(fixture_name, request, cv, n_jobs):
    """Test the attributes of the cross-validation report."""
    estimator, X, y = request.getfixturevalue(fixture_name)
    report = CrossValidationReport(estimator, X, y, cv=cv, n_jobs=n_jobs)
    assert isinstance(report, CrossValidationReport)
    assert isinstance(report.estimator_reports, list)
    for estimator_report in report.estimator_reports:
        assert isinstance(estimator_report, EstimatorReport)
    assert report.X is X
    assert report.y is y
    assert report.n_jobs == n_jobs
    assert len(report.estimator_reports) == cv
    if isinstance(estimator, Pipeline):
        assert report.estimator_name == estimator[-1].__class__.__name__
    else:
        assert report.estimator_name == estimator.__class__.__name__

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
@pytest.mark.parametrize("n_jobs", [None, 1, 2])
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

    _check_metrics_names(result, [metric], expected_nb_stats)

    # check the aggregate parameter
    stats = ["mean", "std"]
    result = getattr(report.metrics, metric)(aggregate=stats)
    # check that the index contains the expected split names
    split_names = result.index.get_level_values(1).unique()
    assert list(split_names) == stats

    stats = "mean"
    result = getattr(report.metrics, metric)(aggregate=stats)
    # check that the index contains the expected split names
    split_names = result.index.get_level_values(1).unique()
    assert list(split_names) == [stats]


def _check_results_report_metric(
    report, params, expected_n_splits, expected_metrics, expected_nb_stats
):
    result = report.metrics.report_metrics(**params)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == expected_n_splits
    # check that we hit the cache
    result_with_cache = report.metrics.report_metrics(**params)
    pd.testing.assert_frame_equal(result, result_with_cache)

    # check that the index contains the expected split names
    split_names = result.index.get_level_values(1).unique()
    expected_split_names = [f"Split #{i}" for i in range(expected_n_splits)]
    assert list(split_names) == expected_split_names

    _check_metrics_names(result, expected_metrics, expected_nb_stats)

    # check the aggregate parameter
    stats = ["mean", "std"]
    result = report.metrics.report_metrics(aggregate=stats, **params)
    # check that the index contains the expected split names
    split_names = result.index.get_level_values(1).unique()
    assert list(split_names) == stats

    stats = "mean"
    result = report.metrics.report_metrics(aggregate=stats, **params)
    # check that the index contains the expected split names
    split_names = result.index.get_level_values(1).unique()
    assert list(split_names) == [stats]


def _check_metrics_names(result, expected_metrics, expected_nb_stats):
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


@pytest.mark.parametrize("pos_label, nb_stats", [(None, 2), (1, 1)])
def test_cross_validation_report_report_metrics_binary(
    binary_classification_data,
    binary_classification_data_svc,
    pos_label,
    nb_stats,
):
    """Check the behaviour of the `report_metrics` method with binary
    classification. We test both with an SVC that does not support `predict_proba` and a
    RandomForestClassifier that does.
    """
    estimator, X, y = binary_classification_data
    report = CrossValidationReport(estimator, X, y, cv=2)
    expected_metrics = ("precision", "recall", "roc_auc", "brier_score")
    # depending on `pos_label`, we report a stats for each class or not for
    # precision and recall
    expected_nb_stats = 2 * nb_stats + 2
    _check_results_report_metric(
        report,
        params={"pos_label": pos_label},
        expected_n_splits=2,
        expected_metrics=expected_metrics,
        expected_nb_stats=expected_nb_stats,
    )

    # Repeat the same experiment where we the target labels are not [0, 1] but
    # ["neg", "pos"]. We check that we don't get any error.
    target_names = np.array(["neg", "pos"], dtype=object)
    pos_label_name = target_names[pos_label] if pos_label is not None else pos_label
    y = target_names[y]
    report = CrossValidationReport(estimator, X, y, cv=2)
    expected_metrics = ("precision", "recall", "roc_auc", "brier_score")
    # depending on `pos_label`, we report a stats for each class or not for
    # precision and recall
    expected_nb_stats = 2 * nb_stats + 2
    _check_results_report_metric(
        report,
        params={"pos_label": pos_label_name},
        expected_n_splits=2,
        expected_metrics=expected_metrics,
        expected_nb_stats=expected_nb_stats,
    )

    estimator, X, y = binary_classification_data_svc
    report = CrossValidationReport(estimator, X, y, cv=2)
    expected_metrics = ("precision", "recall", "roc_auc")
    # depending on `pos_label`, we report a stats for each class or not for
    # precision and recall
    expected_nb_stats = 2 * nb_stats + 1
    _check_results_report_metric(
        report,
        params={"pos_label": pos_label},
        expected_n_splits=2,
        expected_metrics=expected_metrics,
        expected_nb_stats=expected_nb_stats,
    )


def test_cross_validation_report_report_metrics_multiclass(
    multiclass_classification_data, multiclass_classification_data_svc
):
    """Check the behaviour of the `report_metrics` method with multiclass
    classification.
    """
    estimator, X, y = multiclass_classification_data
    report = CrossValidationReport(estimator, X, y, cv=2)
    expected_metrics = ("precision", "recall", "roc_auc", "log_loss")
    # since we are not averaging by default, we report 3 statistics for
    # precision, recall and roc_auc
    expected_nb_stats = 3 * 3 + 1
    _check_results_report_metric(
        report,
        params={},
        expected_n_splits=2,
        expected_metrics=expected_metrics,
        expected_nb_stats=expected_nb_stats,
    )

    estimator, X, y = multiclass_classification_data_svc
    report = CrossValidationReport(estimator, X, y, cv=2)
    expected_metrics = ("precision", "recall")
    # since we are not averaging by default, we report 3 statistics for
    # precision and recall
    expected_nb_stats = 3 * 2
    _check_results_report_metric(
        report,
        params={},
        expected_n_splits=2,
        expected_metrics=expected_metrics,
        expected_nb_stats=expected_nb_stats,
    )


def test_cross_validation_report_report_metrics_regression(regression_data):
    """Check the behaviour of the `report_metrics` method with regression."""
    estimator, X, y = regression_data
    report = CrossValidationReport(estimator, X, y, cv=2)
    expected_metrics = ("r2", "rmse")
    _check_results_report_metric(
        report,
        params={},
        expected_n_splits=2,
        expected_metrics=expected_metrics,
        expected_nb_stats=len(expected_metrics),
    )


def test_cross_validation_report_report_metrics_scoring_kwargs(
    regression_multioutput_data, multiclass_classification_data
):
    """Check the behaviour of the `report_metrics` method with scoring kwargs."""
    estimator, X, y = regression_multioutput_data
    report = CrossValidationReport(estimator, X, y, cv=2)
    assert hasattr(report.metrics, "report_metrics")
    result = report.metrics.report_metrics(scoring_kwargs={"multioutput": "raw_values"})
    assert result.shape == (2, 4)
    assert isinstance(result.columns, pd.MultiIndex)
    assert result.columns.names == ["Metric", "Output"]

    estimator, X, y = multiclass_classification_data
    report = CrossValidationReport(estimator, X, y, cv=2)
    assert hasattr(report.metrics, "report_metrics")
    result = report.metrics.report_metrics(scoring_kwargs={"average": None})
    assert result.shape == (2, 10)
    assert isinstance(result.columns, pd.MultiIndex)
    assert result.columns.names == ["Metric", "Class label"]


@pytest.mark.parametrize(
    "fixture_name, scoring_names, expected_columns",
    [
        ("regression_data", ["R2", "RMSE"], ["R2", "RMSE"]),
        (
            "multiclass_classification_data",
            ["Precision", "Recall", "ROC AUC", "Log Loss"],
            [
                "Precision",
                "Precision",
                "Precision",
                "Recall",
                "Recall",
                "Recall",
                "ROC AUC",
                "ROC AUC",
                "ROC AUC",
                "Log Loss",
            ],
        ),
    ],
)
def test_cross_validation_report_report_metrics_overwrite_scoring_names(
    request, fixture_name, scoring_names, expected_columns
):
    """Test that we can overwrite the scoring names in report_metrics."""
    estimator, X, y = request.getfixturevalue(fixture_name)
    report = CrossValidationReport(estimator, X, y, cv=2)
    result = report.metrics.report_metrics(scoring_names=scoring_names)
    assert result.shape == (2, len(expected_columns))

    # Get level 0 names if MultiIndex, otherwise get column names
    result_columns = (
        result.columns.get_level_values(0).tolist()
        if isinstance(result.columns, pd.MultiIndex)
        else result.columns.tolist()
    )
    assert result_columns == expected_columns


@pytest.mark.parametrize("scoring", ["public_metric", "_private_metric"])
def test_cross_validation_report_report_metrics_error_scoring_strings(
    regression_data, scoring
):
    """Check that we raise an error if a scoring string is not a valid metric."""
    estimator, X, y = regression_data
    report = CrossValidationReport(estimator, X, y, cv=2)
    err_msg = re.escape(f"Invalid metric: {scoring!r}.")
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.report_metrics(scoring=[scoring])
