import re

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from skore import EstimatorReport
from skore.sklearn._estimator import _check_supported_estimator
from skore.sklearn._plot import RocCurveDisplay


@pytest.fixture
def binary_classification_data():
    """Create a binary classification dataset and return fitted estimator and data."""
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return RandomForestClassifier().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def multiclass_classification_data():
    """Create a multiclass classification dataset and return fitted estimator and
    data."""
    X, y = make_classification(
        n_classes=3, n_clusters_per_class=1, random_state=42, n_informative=10
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return RandomForestClassifier().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def regression_data():
    """Create a regression dataset and return fitted estimator and data."""
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return LinearRegression().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def regression_multioutput_data():
    """Create a regression dataset and return fitted estimator and data."""
    X, y = make_regression(n_targets=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return LinearRegression().fit(X_train, y_train), X_test, y_test


def test_check_supported_estimator():
    """Test the behaviour of `_check_supported_estimator`."""

    class MockParent:
        def __init__(self, estimator):
            self.estimator = estimator

    class MockAccessor:
        def __init__(self, parent):
            self._parent = parent

    parent = MockParent(LogisticRegression())
    accessor = MockAccessor(parent)
    check = _check_supported_estimator((LogisticRegression,))
    assert check(accessor)

    pipeline = Pipeline([("clf", LogisticRegression())])
    parent = MockParent(pipeline)
    accessor = MockAccessor(parent)
    assert check(accessor)

    parent = MockParent(RandomForestClassifier())
    accessor = MockAccessor(parent)
    err_msg = (
        "The RandomForestClassifier estimator is not supported by the function called."
    )
    with pytest.raises(AttributeError, match=err_msg):
        check(accessor)


########################################################################################
# Check the general behaviour of the report
########################################################################################


def test_estimator_not_fitted():
    """Test that an error is raised when trying to create a report from an unfitted
    estimator.
    """
    estimator = LinearRegression()
    with pytest.raises(NotFittedError):
        EstimatorReport.from_fitted_estimator(estimator, X=None, y=None)


def test_estimator_report_from_unfitted_estimator():
    """Check the general behaviour of `from_unfitted_estimator`."""
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator = LinearRegression()
    report = EstimatorReport.from_unfitted_estimator(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    check_is_fitted(report.estimator)
    assert report.estimator is not estimator  # the estimator should be cloned

    assert report.X_train is X_train
    assert report.y_train is y_train
    assert report.X_val is X_test
    assert report.y_val is y_test

    err_msg = "attribute is immutable"
    with pytest.raises(AttributeError, match=err_msg):
        report.estimator = LinearRegression()
    with pytest.raises(AttributeError, match=err_msg):
        report.X_train = X_train
    with pytest.raises(AttributeError, match=err_msg):
        report.y_train = y_train


def test_estimator_report_from_fitted_estimator(binary_classification_data):
    """Check the general behaviour of `from_fitted_estimator`."""
    estimator, X, y = binary_classification_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)

    assert report.estimator is estimator  # we should not clone the estimator
    assert report.X_train is None
    assert report.y_train is None
    assert report.X_val is X
    assert report.y_val is y

    err_msg = "attribute is immutable"
    with pytest.raises(AttributeError, match=err_msg):
        report.estimator = LinearRegression()
    with pytest.raises(AttributeError, match=err_msg):
        report.X_train = X
    with pytest.raises(AttributeError, match=err_msg):
        report.y_train = y


def test_estimator_report_invalidate_cache_data(binary_classification_data):
    """Check that we invalidate the cache when the data is changed."""
    estimator, X, y = binary_classification_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)

    for attribute in ("X_val", "y_val"):
        report._cache["mocking"] = "mocking"  # mock writing to cache
        setattr(report, attribute, None)
        assert report._cache == {}


@pytest.mark.parametrize(
    "X, y, supported_plot_methods, not_supported_plot_methods",
    [
        (*make_classification(random_state=42), ["roc"], []),
        (
            *make_classification(n_classes=3, n_clusters_per_class=1, random_state=42),
            [],
            ["roc"],
        ),
    ],
)
def test_estimator_report_check_support_plot(
    X, y, supported_plot_methods, not_supported_plot_methods
):
    """Check that the available plot methods are correctly registered."""
    classifier = RandomForestClassifier().fit(X, y)
    report = EstimatorReport.from_fitted_estimator(classifier, X=X, y=y)

    for supported_plot_method in supported_plot_methods:
        assert hasattr(report.plot, supported_plot_method)

    for not_supported_plot_method in not_supported_plot_methods:
        assert not hasattr(report.plot, not_supported_plot_method)


def test_estimator_report_help(capsys, binary_classification_data):
    """Check that the help method writes to the console."""
    estimator, X, y = binary_classification_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)

    report.help()
    captured = capsys.readouterr()
    assert (
        f"🔧 Available tools for diagnosing {estimator.__class__.__name__} estimator"
        in captured.out
    )


def test_estimator_report_repr(binary_classification_data):
    """Check that __repr__ returns a string starting with the expected prefix."""
    estimator, X, y = binary_classification_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)

    repr_str = repr(report)
    assert repr_str.startswith("📓 Estimator Reporter")


########################################################################################
# Check the plot methods
########################################################################################


def test_estimator_report_plot_roc(binary_classification_data):
    """Check that the ROC plot method works."""
    estimator, X, y = binary_classification_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)
    assert isinstance(report.plot.roc(), RocCurveDisplay)


########################################################################################
# Check the metrics methods
########################################################################################


@pytest.mark.parametrize(
    "metric", ["accuracy", "precision", "recall", "brier_score", "roc_auc", "log_loss"]
)
def test_estimator_report_metrics_binary_classification(
    binary_classification_data, metric
):
    """Check the behaviour of the metrics methods available for binary
    classification.
    """
    estimator, X, y = binary_classification_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)
    assert hasattr(report.metrics, metric)
    result = getattr(report.metrics, metric)()
    assert isinstance(result, pd.DataFrame)

    # check that something was written to the cache
    assert report._cache != {}
    report.clean_cache()

    # check that passing using data outside from the report works and that we they
    # don't come from the cache
    result_external_data = getattr(report.metrics, metric)(X=X, y=y)
    assert isinstance(result_external_data, pd.DataFrame)
    pd.testing.assert_frame_equal(result, result_external_data)
    assert report._cache == {}


@pytest.mark.parametrize("metric", ["r2", "rmse"])
def test_estimator_report_metrics_regression(regression_data, metric):
    """Check the behaviour of the metrics methods available for regression."""
    estimator, X, y = regression_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)
    assert hasattr(report.metrics, metric)
    result = getattr(report.metrics, metric)()
    assert isinstance(result, pd.DataFrame)

    # check that something was written to the cache
    assert report._cache != {}
    report.clean_cache()

    # check that passing using data outside from the report works and that we they
    # don't come from the cache
    result_external_data = getattr(report.metrics, metric)(X=X, y=y)
    assert isinstance(result_external_data, pd.DataFrame)
    pd.testing.assert_frame_equal(result, result_external_data)
    assert report._cache == {}


def test_estimator_report_report_metrics_binary(binary_classification_data):
    """Check the behaviour of the `report_metrics` method with binary classification."""
    estimator, X, y = binary_classification_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)
    assert hasattr(report.metrics, "report_metrics")
    result = report.metrics.report_metrics()
    assert isinstance(result, pd.DataFrame)
    expected_metrics = ("precision", "recall", "roc_auc", "brier_score")
    assert len(result.columns) == len(expected_metrics)

    def normalize_string(s):
        # Remove spaces, underscores and any non-alphanumeric characters
        return re.sub(r"[^a-zA-Z0-9]", "", s.lower())

    normalized_expected = {normalize_string(metric) for metric in expected_metrics}

    for column in result.columns:
        normalized_column = normalize_string(column)
        matches = [
            metric for metric in normalized_expected if metric == normalized_column
        ]
        assert len(matches) == 1, (
            f"No match found for column '{column}' in expected metrics: "
            f" {expected_metrics}"
        )


def test_estimator_report_report_metrics_multiclass(multiclass_classification_data):
    """Check the behaviour of the `report_metrics` method with multiclass
    classification.
    """
    estimator, X, y = multiclass_classification_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)
    assert hasattr(report.metrics, "report_metrics")
    result = report.metrics.report_metrics()
    assert isinstance(result, pd.DataFrame)
    expected_metrics = ("precision", "recall", "roc_auc", "log_loss")
    assert len(result.columns) == len(expected_metrics)

    def normalize_string(s):
        # Remove spaces, underscores and any non-alphanumeric characters
        return re.sub(r"[^a-zA-Z0-9]", "", s.lower())

    normalized_expected = {normalize_string(metric) for metric in expected_metrics}

    for column in result.columns:
        normalized_column = normalize_string(column)
        matches = [
            metric for metric in normalized_expected if metric == normalized_column
        ]
        assert len(matches) == 1, (
            f"No match found for column '{column}' in expected metrics: "
            f" {expected_metrics}"
        )


def test_estimator_report_report_metrics_regression(regression_data):
    """Check the behaviour of the `report_metrics` method with regression."""
    estimator, X, y = regression_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)
    assert hasattr(report.metrics, "report_metrics")
    result = report.metrics.report_metrics()
    assert isinstance(result, pd.DataFrame)
    expected_metrics = ("r2", "rmse")
    assert len(result.columns) == len(expected_metrics)

    def normalize_string(s):
        # Remove spaces, underscores, numbers and any non-numeric characters
        return re.sub(r"[^a-zA-Z]", "", s.lower())

    normalized_expected = {normalize_string(metric) for metric in expected_metrics}

    for column in result.columns:
        normalized_column = normalize_string(column)
        matches = [
            metric for metric in normalized_expected if metric == normalized_column
        ]
        assert len(matches) == 1, (
            f"No match found for column '{column}' in expected metrics: "
            f" {expected_metrics}"
        )


def test_estimator_report_report_metrics_scoring_kwargs(
    regression_multioutput_data, multiclass_classification_data
):
    """Check the behaviour of the `report_metrics` method with scoring kwargs."""
    estimator, X, y = regression_multioutput_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)
    assert hasattr(report.metrics, "report_metrics")
    result = report.metrics.report_metrics(scoring_kwargs={"multioutput": "raw_values"})
    assert result.shape == (1, 4)
    assert isinstance(result.columns, pd.MultiIndex)
    assert result.columns.names == ["Metric", "Output"]

    estimator, X, y = multiclass_classification_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)
    assert hasattr(report.metrics, "report_metrics")
    result = report.metrics.report_metrics(scoring_kwargs={"average": None})
    assert result.shape == (1, 10)
    assert isinstance(result.columns, pd.MultiIndex)
    assert result.columns.names == ["Metric", "Class label"]


def test_estimator_report_interaction_cache_metrics(regression_multioutput_data):
    """Check that the cache take into account the 'kwargs' of a metric."""
    estimator, X, y = regression_multioutput_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)

    # The underlying metrics will call `_compute_metric_scores` that take some arbitrary
    # kwargs apart from `pos_label`. Let's pass an arbitrary kwarg and make sure it is
    # part of the cache.
    multioutput = "raw_values"
    result_r2_raw_values = report.metrics.r2(multioutput=multioutput)
    should_raise = True
    for cached_key in report._cache:
        if any(item == multioutput for item in cached_key):
            should_raise = False
            break
    assert (
        not should_raise
    ), f"The value {multioutput} should be stored in one of the cache keys"
    assert result_r2_raw_values.shape == (1, 2)

    multioutput = "uniform_average"
    result_r2_uniform_average = report.metrics.r2(multioutput=multioutput)
    should_raise = True
    for cached_key in report._cache:
        if any(item == multioutput for item in cached_key):
            should_raise = False
            break
    assert (
        not should_raise
    ), f"The value {multioutput} should be stored in one of the cache keys"
    assert result_r2_uniform_average.shape == (1, 1)


def test_estimator_report_custom_metric(regression_data):
    """Check the behaviour of the `custom_metric` computation in the report."""
    estimator, X, y = regression_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)

    def custom_metric(y_true, y_pred, threshold=0.5):
        residuals = y_true - y_pred
        return np.mean(np.where(residuals < threshold, residuals, 1))

    threshold = 1
    result = report.metrics.custom_metric(
        metric_function=custom_metric,
        metric_name="Custom Metric",
        response_method="predict",
        threshold=threshold,
    )
    should_raise = True
    for cached_key in report._cache:
        if any(item == threshold for item in cached_key):
            should_raise = False
            break
    assert (
        not should_raise
    ), f"The value {threshold} should be stored in one of the cache keys"

    assert result.columns.tolist() == ["Custom Metric"]
    assert result.to_numpy()[0, 0] == pytest.approx(
        custom_metric(y, estimator.predict(X), threshold)
    )

    threshold = 100
    result = report.metrics.custom_metric(
        metric_function=custom_metric,
        metric_name="Custom Metric",
        response_method="predict",
        threshold=threshold,
    )
    should_raise = True
    for cached_key in report._cache:
        if any(item == threshold for item in cached_key):
            should_raise = False
            break
    assert (
        not should_raise
    ), f"The value {threshold} should be stored in one of the cache keys"

    assert result.columns.tolist() == ["Custom Metric"]
    assert result.to_numpy()[0, 0] == pytest.approx(
        custom_metric(y, estimator.predict(X), threshold)
    )


def test_estimator_report_custom_function_kwargs_numpy_array(regression_data):
    """Check that we are able to store a hash of a numpy array in the cache when they
    are passed as kwargs.
    """
    estimator, X, y = regression_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)
    weights = np.ones_like(y) * 2
    hash_weights = joblib.hash(weights)

    def custom_metric(y_true, y_pred, some_weights):
        return np.mean((y_true - y_pred) * some_weights)

    result = report.metrics.custom_metric(
        metric_function=custom_metric,
        metric_name="Custom Metric",
        response_method="predict",
        some_weights=weights,
    )
    should_raise = True
    for cached_key in report._cache:
        if any(item == hash_weights for item in cached_key):
            should_raise = False
            break
    assert (
        not should_raise
    ), "The hash of the weights should be stored in one of the cache keys"

    assert result.columns.tolist() == ["Custom Metric"]
    assert result.to_numpy()[0, 0] == pytest.approx(
        custom_metric(y, estimator.predict(X), weights)
    )


def test_estimator_report_report_metrics_with_custom_metric(regression_data):
    """Check that we can pass a custom metric with specific kwargs into
    `report_metrics`."""
    estimator, X, y = regression_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)
    weights = np.ones_like(y) * 2

    def custom_metric(y_true, y_pred, some_weights):
        return np.mean((y_true - y_pred) * some_weights)

    result = report.metrics.report_metrics(
        scoring=["r2", custom_metric],
        scoring_kwargs={"some_weights": weights, "response_method": "predict"},
    )
    assert result.shape == (1, 2)
    np.testing.assert_allclose(
        result.to_numpy(),
        [
            [
                r2_score(y, estimator.predict(X)),
                custom_metric(y, estimator.predict(X), weights),
            ]
        ],
    )


def test_estimator_report_report_metrics_with_scorer(regression_data):
    """Check that we can pass scikit-learn scorer with different parameters to
    the `report_metrics` method."""
    estimator, X, y = regression_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)
    weights = np.ones_like(y) * 2

    def custom_metric(y_true, y_pred, some_weights):
        return np.mean((y_true - y_pred) * some_weights)

    median_absolute_error_scorer = make_scorer(
        median_absolute_error, response_method="predict"
    )
    custom_metric_scorer = make_scorer(
        custom_metric, response_method="predict", some_weights=weights
    )
    result = report.metrics.report_metrics(
        scoring=[r2_score, median_absolute_error_scorer, custom_metric_scorer],
        scoring_kwargs={"response_method": "predict"},  # only dispatched to r2_score
    )
    assert result.shape == (1, 3)
    np.testing.assert_allclose(
        result.to_numpy(),
        [
            [
                r2_score(y, estimator.predict(X)),
                median_absolute_error(y, estimator.predict(X)),
                custom_metric(y, estimator.predict(X), weights),
            ]
        ],
    )
