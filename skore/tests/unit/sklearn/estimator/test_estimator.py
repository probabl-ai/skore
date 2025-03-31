import re
from copy import deepcopy
from io import BytesIO
from numbers import Real

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    median_absolute_error,
    r2_score,
    rand_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from skore import EstimatorReport
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
def binary_classification_data_svc():
    """Create a binary classification dataset and return fitted estimator and data.
    The estimator is a SVC that does not support `predict_proba`.
    """
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return SVC().fit(X_train, y_train), X_test, y_test


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
def multiclass_classification_data_svc():
    """Create a multiclass classification dataset and return fitted estimator and
    data. The estimator is a SVC that does not support `predict_proba`.
    """
    X, y = make_classification(
        n_classes=3, n_clusters_per_class=1, random_state=42, n_informative=10
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return SVC().fit(X_train, y_train), X_test, y_test


@pytest.fixture
def binary_classification_data_pipeline():
    """Create a binary classification dataset and return fitted pipeline and data."""
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    estimator = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    return estimator.fit(X_train, y_train), X_test, y_test


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


########################################################################################
# Check the general behaviour of the report
########################################################################################


@pytest.mark.parametrize("fit", [True, "auto"])
def test_estimator_not_fitted(fit):
    """Test that an error is raised when trying to create a report from an unfitted
    estimator and no data are provided to fit the estimator.
    """
    estimator = LinearRegression()
    err_msg = "The training data is required to fit the estimator. "
    with pytest.raises(ValueError, match=err_msg):
        EstimatorReport(estimator, fit=fit)


@pytest.mark.parametrize("fit", [True, "auto"])
def test_estimator_report_from_unfitted_estimator(fit):
    """Check the general behaviour of passing an unfitted estimator and training
    data."""
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator = LinearRegression()
    report = EstimatorReport(
        estimator,
        fit=fit,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    check_is_fitted(report.estimator_)
    assert report.estimator_ is not estimator  # the estimator should be cloned

    assert report.X_train is X_train
    assert report.y_train is y_train
    assert report.X_test is X_test
    assert report.y_test is y_test

    err_msg = "attribute is immutable"
    with pytest.raises(AttributeError, match=err_msg):
        report.estimator_ = LinearRegression()
    with pytest.raises(AttributeError, match=err_msg):
        report.X_train = X_train
    with pytest.raises(AttributeError, match=err_msg):
        report.y_train = y_train


@pytest.mark.parametrize("fit", [False, "auto"])
def test_estimator_report_from_fitted_estimator(binary_classification_data, fit):
    """Check the general behaviour of passing an already fitted estimator without
    refitting it."""
    estimator, X, y = binary_classification_data
    report = EstimatorReport(estimator, fit=fit, X_test=X, y_test=y)

    check_is_fitted(report.estimator_)
    assert isinstance(report.estimator_, RandomForestClassifier)
    assert report.X_train is None
    assert report.y_train is None
    assert report.X_test is X
    assert report.y_test is y

    err_msg = "attribute is immutable"
    with pytest.raises(AttributeError, match=err_msg):
        report.estimator_ = LinearRegression()
    with pytest.raises(AttributeError, match=err_msg):
        report.X_train = X
    with pytest.raises(AttributeError, match=err_msg):
        report.y_train = y


def test_estimator_report_from_fitted_pipeline(binary_classification_data_pipeline):
    """Check the general behaviour of passing an already fitted pipeline without
    refitting it.
    """
    estimator, X, y = binary_classification_data_pipeline
    report = EstimatorReport(estimator, X_test=X, y_test=y)

    check_is_fitted(report.estimator_)
    assert isinstance(report.estimator_, Pipeline)
    assert report.estimator_name_ == estimator[-1].__class__.__name__
    assert report.X_train is None
    assert report.y_train is None
    assert report.X_test is X
    assert report.y_test is y


def test_estimator_report_invalidate_cache_data(binary_classification_data):
    """Check that we invalidate the cache when the data is changed."""
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    for attribute in ("X_test", "y_test"):
        report._cache["mocking"] = "mocking"  # mock writing to cache
        setattr(report, attribute, None)
        assert report._cache == {}


@pytest.mark.parametrize(
    "Estimator, X_test, y_test, supported_plot_methods, not_supported_plot_methods",
    [
        (
            RandomForestClassifier(),
            *make_classification(random_state=42),
            ["roc", "precision_recall"],
            ["prediction_error"],
        ),
        (
            RandomForestClassifier(),
            *make_classification(n_classes=3, n_clusters_per_class=1, random_state=42),
            ["roc", "precision_recall"],
            ["prediction_error"],
        ),
        (
            LinearRegression(),
            *make_regression(random_state=42),
            ["prediction_error"],
            ["roc", "precision_recall"],
        ),
    ],
)
def test_estimator_report_check_support_plot(
    Estimator, X_test, y_test, supported_plot_methods, not_supported_plot_methods
):
    """Check that the available plot methods are correctly registered."""
    estimator = Estimator.fit(X_test, y_test)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    for supported_plot_method in supported_plot_methods:
        assert hasattr(report.metrics, supported_plot_method)

    for not_supported_plot_method in not_supported_plot_methods:
        assert not hasattr(report.metrics, not_supported_plot_method)


def test_estimator_report_help(capsys, binary_classification_data):
    """Check that the help method writes to the console."""
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    report.help()
    captured = capsys.readouterr()
    assert f"Tools to diagnose estimator {estimator.__class__.__name__}" in captured.out

    # Check that we have a line with accuracy and the arrow associated with it
    assert re.search(
        r"\.accuracy\([^)]*\).*\(↗︎\).*-.*accuracy", captured.out, re.MULTILINE
    )


def test_estimator_report_repr(binary_classification_data):
    """Check that __repr__ returns a string starting with the expected prefix."""
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    repr_str = repr(report)
    assert "EstimatorReport" in repr_str


@pytest.mark.parametrize(
    "fixture_name, pass_train_data, expected_n_keys",
    [
        ("binary_classification_data", True, 10),
        ("binary_classification_data_svc", True, 10),
        ("multiclass_classification_data", True, 12),
        ("regression_data", True, 4),
        ("binary_classification_data", False, 5),
        ("binary_classification_data_svc", False, 5),
        ("multiclass_classification_data", False, 6),
        ("regression_data", False, 2),
    ],
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_estimator_report_cache_predictions(
    request, fixture_name, pass_train_data, expected_n_keys, n_jobs
):
    """Check that calling cache_predictions fills the cache."""
    estimator, X_test, y_test = request.getfixturevalue(fixture_name)
    if pass_train_data:
        report = EstimatorReport(
            estimator, X_train=X_test, y_train=y_test, X_test=X_test, y_test=y_test
        )
    else:
        report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    assert report._cache == {}
    report.cache_predictions(n_jobs=n_jobs)
    assert len(report._cache) == expected_n_keys
    assert report._cache != {}
    stored_cache = deepcopy(report._cache)
    report.cache_predictions(n_jobs=n_jobs)
    # check that the keys are exactly the same
    assert report._cache.keys() == stored_cache.keys()


def test_estimator_report_pickle(binary_classification_data):
    """Check that we can pickle an estimator report.

    In particular, the progress bar from rich are pickable, therefore we trigger
    the progress bar to be able to test that the progress bar is pickable.
    """
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    report.cache_predictions()

    with BytesIO() as stream:
        joblib.dump(report, stream)


def test_estimator_report_flat_index(binary_classification_data):
    """Check that the index is flattened when `flat_index` is True.

    Since `pos_label` is None, then by default a MultiIndex would be returned.
    Here, we force to have a single-index by passing `flat_index=True`.
    """
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    result = report.metrics.report_metrics(flat_index=True)
    assert result.shape == (8, 1)
    assert isinstance(result.index, pd.Index)
    assert result.index.tolist() == [
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
        "roc_auc",
        "brier_score",
        "fit_time",
        "predict_time",
    ]
    assert result.columns.tolist() == ["RandomForestClassifier"]


########################################################################################
# Check the plot methods
########################################################################################


def test_estimator_report_plot_roc(binary_classification_data):
    """Check that the ROC plot method works."""
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert isinstance(report.metrics.roc(), RocCurveDisplay)


@pytest.mark.parametrize("display", ["roc", "precision_recall"])
def test_estimator_report_display_binary_classification(
    pyplot, binary_classification_data, display
):
    """The call to display functions should be cached."""
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)()
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)()
    assert display_first_call is display_second_call


@pytest.mark.parametrize("display", ["prediction_error"])
def test_estimator_report_display_regression(pyplot, regression_data, display):
    """The call to display functions should be cached, as long as the arguments make it
    reproducible."""
    estimator, X_test, y_test = regression_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)(random_state=0)
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)(random_state=0)
    assert display_first_call is display_second_call


@pytest.mark.parametrize("display", ["roc", "precision_recall"])
def test_estimator_report_display_binary_classification_external_data(
    pyplot, binary_classification_data, display
):
    """The call to display functions should be cached when passing external data."""
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator)
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)(
        data_source="X_y", X=X_test, y=y_test
    )
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)(
        data_source="X_y", X=X_test, y=y_test
    )
    assert display_first_call is display_second_call


@pytest.mark.parametrize("display", ["prediction_error"])
def test_estimator_report_display_regression_external_data(
    pyplot, regression_data, display
):
    """The call to display functions should be cached when passing external data,
    as long as the arguments make it reproducible."""
    estimator, X_test, y_test = regression_data
    report = EstimatorReport(estimator)
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)(
        data_source="X_y", X=X_test, y=y_test, random_state=0
    )
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)(
        data_source="X_y", X=X_test, y=y_test, random_state=0
    )
    assert display_first_call is display_second_call


@pytest.mark.parametrize("display", ["roc", "precision_recall"])
def test_estimator_report_display_binary_classification_switching_data_source(
    pyplot, binary_classification_data, display
):
    """Check that we don't hit the cache when switching the data source."""
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_test, y_train=y_test, X_test=X_test, y_test=y_test
    )
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)(data_source="test")
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)(data_source="train")
    assert display_first_call is not display_second_call
    display_third_call = getattr(report.metrics, display)(
        data_source="X_y", X=X_test, y=y_test
    )
    assert display_first_call is not display_third_call
    assert display_second_call is not display_third_call


@pytest.mark.parametrize("display", ["prediction_error"])
def test_estimator_report_display_regression_switching_data_source(
    pyplot, regression_data, display
):
    """Check that we don't hit the cache when switching the data source."""
    estimator, X_test, y_test = regression_data
    report = EstimatorReport(
        estimator, X_train=X_test, y_train=y_test, X_test=X_test, y_test=y_test
    )
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)(data_source="test")
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)(data_source="train")
    assert display_first_call is not display_second_call
    display_third_call = getattr(report.metrics, display)(
        data_source="X_y", X=X_test, y=y_test
    )
    assert display_first_call is not display_third_call
    assert display_second_call is not display_third_call


########################################################################################
# Check the metrics methods
########################################################################################


def test_estimator_report_metrics_help(capsys, binary_classification_data):
    """Check that the help method writes to the console."""
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    report.metrics.help()
    captured = capsys.readouterr()
    assert "Available metrics methods" in captured.out


def test_estimator_report_metrics_repr(binary_classification_data):
    """Check that __repr__ returns a string starting with the expected prefix."""
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    repr_str = repr(report.metrics)
    assert "skore.EstimatorReport.metrics" in repr_str
    assert "help()" in repr_str


@pytest.mark.parametrize("metric", ["accuracy", "brier_score", "roc_auc", "log_loss"])
def test_estimator_report_metrics_binary_classification(
    binary_classification_data, metric
):
    """Check the behaviour of the metrics methods available for binary
    classification.
    """
    estimator, X_test, y_test = binary_classification_data
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
def test_estimator_report_metrics_binary_classification_pr(
    binary_classification_data, metric
):
    """Check the behaviour of the precision and recall metrics available for binary
    classification.
    """
    estimator, X_test, y_test = binary_classification_data
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
def test_estimator_report_metrics_regression(regression_data, metric):
    """Check the behaviour of the metrics methods available for regression."""
    estimator, X_test, y_test = regression_data
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


def _normalize_metric_name(column):
    """Helper to normalize the metric name present in a pandas column that could be
    a multi-index or single-index."""
    # if we have a multi-index, then the metric name is on level 0
    s = column[0] if isinstance(column, tuple) else column
    # Remove spaces and underscores
    return re.sub(r"[^a-zA-Z]", "", s.lower())


def _check_results_report_metrics(result, expected_metrics, expected_nb_stats):
    assert isinstance(result, pd.DataFrame)
    assert len(result.index) == expected_nb_stats

    normalized_expected = {
        _normalize_metric_name(metric) for metric in expected_metrics
    }
    for column in result.index:
        normalized_column = _normalize_metric_name(column)
        matches = [
            metric for metric in normalized_expected if metric == normalized_column
        ]
        assert len(matches) == 1, (
            f"No match found for column '{column}' in expected metrics: "
            f" {expected_metrics}"
        )


@pytest.mark.parametrize("pos_label, nb_stats", [(None, 2), (1, 1)])
@pytest.mark.parametrize("data_source", ["test", "X_y"])
def test_estimator_report_report_metrics_binary(
    binary_classification_data,
    binary_classification_data_svc,
    pos_label,
    nb_stats,
    data_source,
):
    """Check the behaviour of the `report_metrics` method with binary
    classification. We test both with an SVC that does not support `predict_proba` and a
    RandomForestClassifier that does.
    """
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    result = report.metrics.report_metrics(
        pos_label=pos_label, data_source=data_source, **kwargs
    )
    assert "Favorability" not in result.columns
    expected_metrics = (
        "precision",
        "recall",
        "roc_auc",
        "brier_score",
        "fit_time",
        "predict_time",
    )
    # depending on `pos_label`, we report a stats for each class or not for
    # precision and recall
    expected_nb_stats = 2 * nb_stats + 4
    _check_results_report_metrics(result, expected_metrics, expected_nb_stats)

    # Repeat the same experiment where we the target labels are not [0, 1] but
    # ["neg", "pos"]. We check that we don't get any error.
    target_names = np.array(["neg", "pos"], dtype=object)
    pos_label_name = target_names[pos_label] if pos_label is not None else pos_label
    y_test = target_names[y_test]
    estimator = clone(estimator).fit(X_test, y_test)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    result = report.metrics.report_metrics(
        pos_label=pos_label_name, data_source=data_source, **kwargs
    )
    expected_metrics = (
        "precision",
        "recall",
        "roc_auc",
        "brier_score",
        "fit_time",
        "predict_time",
    )
    # depending on `pos_label`, we report a stats for each class or not for
    # precision and recall
    expected_nb_stats = 2 * nb_stats + 4
    _check_results_report_metrics(result, expected_metrics, expected_nb_stats)

    estimator, X_test, y_test = binary_classification_data_svc
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    result = report.metrics.report_metrics(
        pos_label=pos_label, data_source=data_source, **kwargs
    )
    expected_metrics = ("precision", "recall", "roc_auc", "fit_time", "predict_time")
    # depending on `pos_label`, we report a stats for each class or not for
    # precision and recall
    expected_nb_stats = 2 * nb_stats + 3
    _check_results_report_metrics(result, expected_metrics, expected_nb_stats)


@pytest.mark.parametrize("data_source", ["test", "X_y"])
def test_estimator_report_report_metrics_multiclass(
    multiclass_classification_data, multiclass_classification_data_svc, data_source
):
    """Check the behaviour of the `report_metrics` method with multiclass
    classification.
    """
    estimator, X_test, y_test = multiclass_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    result = report.metrics.report_metrics(data_source=data_source, **kwargs)
    assert "Favorability" not in result.columns
    expected_metrics = (
        "precision",
        "recall",
        "roc_auc",
        "log_loss",
        "fit_time",
        "predict_time",
    )
    # since we are not averaging by default, we report 3 statistics for
    # precision, recall and roc_auc
    expected_nb_stats = 3 * 3 + 3
    _check_results_report_metrics(result, expected_metrics, expected_nb_stats)

    estimator, X_test, y_test = multiclass_classification_data_svc
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    result = report.metrics.report_metrics(data_source=data_source, **kwargs)
    expected_metrics = ("precision", "recall", "fit_time", "predict_time")
    # since we are not averaging by default, we report 3 statistics for
    # precision and recall
    expected_nb_stats = 3 * 2 + 2
    _check_results_report_metrics(result, expected_metrics, expected_nb_stats)


@pytest.mark.parametrize("data_source", ["test", "X_y"])
def test_estimator_report_report_metrics_regression(regression_data, data_source):
    """Check the behaviour of the `report_metrics` method with regression."""
    estimator, X_test, y_test = regression_data
    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    result = report.metrics.report_metrics(data_source=data_source, **kwargs)
    assert "Favorability" not in result.columns
    expected_metrics = ("r2", "rmse", "fit_time", "predict_time")
    _check_results_report_metrics(result, expected_metrics, len(expected_metrics))


def test_estimator_report_report_metrics_scoring_kwargs(
    regression_multioutput_data, multiclass_classification_data
):
    """Check the behaviour of the `report_metrics` method with scoring kwargs."""
    estimator, X_test, y_test = regression_multioutput_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, "report_metrics")
    result = report.metrics.report_metrics(scoring_kwargs={"multioutput": "raw_values"})
    assert result.shape == (6, 1)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["Metric", "Output"]

    estimator, X_test, y_test = multiclass_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, "report_metrics")
    result = report.metrics.report_metrics(scoring_kwargs={"average": None})
    assert result.shape == (12, 1)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["Metric", "Label / Average"]


@pytest.mark.parametrize(
    "fixture_name, scoring_names, expected_columns",
    [
        (
            "regression_data",
            ["R2", "RMSE", "FIT_TIME", "PREDICT_TIME"],
            ["R2", "RMSE", "FIT_TIME", "PREDICT_TIME"],
        ),
        (
            "multiclass_classification_data",
            ["Precision", "Recall", "ROC AUC", "Log Loss", "Fit Time", "Predict Time"],
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
                "Fit Time",
                "Predict Time",
            ],
        ),
    ],
)
def test_estimator_report_report_metrics_overwrite_scoring_names(
    request, fixture_name, scoring_names, expected_columns
):
    """Test that we can overwrite the scoring names in report_metrics."""
    estimator, X_test, y_test = request.getfixturevalue(fixture_name)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    result = report.metrics.report_metrics(scoring_names=scoring_names)
    assert result.shape == (len(expected_columns), 1)

    # Get level 0 names if MultiIndex, otherwise get column names
    result_index = (
        result.index.get_level_values(0).tolist()
        if isinstance(result.index, pd.MultiIndex)
        else result.index.tolist()
    )
    assert result_index == expected_columns


def test_estimator_report_report_metrics_indicator_favorability(
    binary_classification_data,
):
    """Check that the behaviour of `indicator_favorability` is correct."""
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    result = report.metrics.report_metrics(indicator_favorability=True)
    assert "Favorability" in result.columns
    indicator = result["Favorability"]
    assert indicator["Precision"].tolist() == ["(↗︎)", "(↗︎)"]
    assert indicator["Recall"].tolist() == ["(↗︎)", "(↗︎)"]
    assert indicator["ROC AUC"].tolist() == ["(↗︎)"]
    assert indicator["Brier score"].tolist() == ["(↘︎)"]


def test_estimator_report_interaction_cache_metrics(regression_multioutput_data):
    """Check that the cache take into account the 'kwargs' of a metric."""
    estimator, X_test, y_test = regression_multioutput_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

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
    assert not should_raise, (
        f"The value {multioutput} should be stored in one of the cache keys"
    )
    assert len(result_r2_raw_values) == 2

    multioutput = "uniform_average"
    result_r2_uniform_average = report.metrics.r2(multioutput=multioutput)
    should_raise = True
    for cached_key in report._cache:
        if any(item == multioutput for item in cached_key):
            should_raise = False
            break
    assert not should_raise, (
        f"The value {multioutput} should be stored in one of the cache keys"
    )
    assert isinstance(result_r2_uniform_average, float)


def test_estimator_report_custom_metric(regression_data):
    """Check the behaviour of the `custom_metric` computation in the report."""
    estimator, X_test, y_test = regression_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def custom_metric(y_true, y_pred, threshold=0.5):
        residuals = y_true - y_pred
        return np.mean(np.where(residuals < threshold, residuals, 1))

    threshold = 1
    result = report.metrics.custom_metric(
        metric_function=custom_metric,
        response_method="predict",
        threshold=threshold,
    )
    should_raise = True
    for cached_key in report._cache:
        if any(item == threshold for item in cached_key):
            should_raise = False
            break
    assert not should_raise, (
        f"The value {threshold} should be stored in one of the cache keys"
    )

    assert isinstance(result, float)
    assert result == pytest.approx(
        custom_metric(y_test, estimator.predict(X_test), threshold)
    )

    threshold = 100
    result = report.metrics.custom_metric(
        metric_function=custom_metric,
        response_method="predict",
        threshold=threshold,
    )
    should_raise = True
    for cached_key in report._cache:
        if any(item == threshold for item in cached_key):
            should_raise = False
            break
    assert not should_raise, (
        f"The value {threshold} should be stored in one of the cache keys"
    )

    assert isinstance(result, float)
    assert result == pytest.approx(
        custom_metric(y_test, estimator.predict(X_test), threshold)
    )


@pytest.mark.parametrize("scoring", ["public_metric", "_private_metric"])
def test_estimator_report_report_metrics_error_scoring_strings(
    regression_data, scoring
):
    """Check that we raise an error if a scoring string is not a valid metric."""
    estimator, X_test, y_test = regression_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    err_msg = re.escape(f"Invalid metric: {scoring!r}.")
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.report_metrics(scoring=[scoring])


def test_estimator_report_custom_function_kwargs_numpy_array(regression_data):
    """Check that we are able to store a hash of a numpy array in the cache when they
    are passed as kwargs.
    """
    estimator, X_test, y_test = regression_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    weights = np.ones_like(y_test) * 2
    hash_weights = joblib.hash(weights)

    def custom_metric(y_true, y_pred, some_weights):
        return np.mean((y_true - y_pred) * some_weights)

    result = report.metrics.custom_metric(
        metric_function=custom_metric,
        response_method="predict",
        some_weights=weights,
    )
    should_raise = True
    for cached_key in report._cache:
        if any(item == hash_weights for item in cached_key):
            should_raise = False
            break
    assert not should_raise, (
        "The hash of the weights should be stored in one of the cache keys"
    )

    assert isinstance(result, float)
    assert result == pytest.approx(
        custom_metric(y_test, estimator.predict(X_test), weights)
    )


def test_estimator_report_report_metrics_with_custom_metric(regression_data):
    """Check that we can pass a custom metric with specific kwargs into
    `report_metrics`."""
    estimator, X_test, y_test = regression_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    weights = np.ones_like(y_test) * 2

    def custom_metric(y_true, y_pred, some_weights):
        return np.mean((y_true - y_pred) * some_weights)

    result = report.metrics.report_metrics(
        scoring=["r2", custom_metric],
        scoring_kwargs={"some_weights": weights, "response_method": "predict"},
    )
    assert result.shape == (2, 1)
    np.testing.assert_allclose(
        result.to_numpy(),
        [
            [r2_score(y_test, estimator.predict(X_test))],
            [custom_metric(y_test, estimator.predict(X_test), weights)],
        ],
    )


def test_estimator_report_report_metrics_with_scorer(regression_data):
    """Check that we can pass scikit-learn scorer with different parameters to
    the `report_metrics` method."""
    estimator, X_test, y_test = regression_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    weights = np.ones_like(y_test) * 2

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
    assert result.shape == (3, 1)
    np.testing.assert_allclose(
        result.to_numpy(),
        [
            [r2_score(y_test, estimator.predict(X_test))],
            [median_absolute_error(y_test, estimator.predict(X_test))],
            [custom_metric(y_test, estimator.predict(X_test), weights)],
        ],
    )


def test_estimator_report_custom_metric_compatible_estimator(
    binary_classification_data,
):
    """Check that the estimator report still works if an estimator has a compatible
    scikit-learn API.
    """
    _, X_test, y_test = binary_classification_data

    class CompatibleEstimator:
        """Estimator exposing only a predict method but it should be enough for the
        reports.
        """

        def fit(self, X, y):
            self.fitted_ = True
            return self

        def predict(self, X):
            return np.ones(X.shape[0])

    estimator = CompatibleEstimator()
    report = EstimatorReport(estimator, fit=False, X_test=X_test, y_test=y_test)
    result = report.metrics.custom_metric(
        metric_function=lambda y_true, y_pred: 1,
        response_method="predict",
    )
    assert isinstance(result, Real)
    assert result == pytest.approx(1)


@pytest.mark.parametrize(
    "scorer, pos_label",
    [
        (
            make_scorer(
                f1_score, response_method="predict", average="macro", pos_label=1
            ),
            1,
        ),
        (
            make_scorer(
                f1_score, response_method="predict", average="macro", pos_label=1
            ),
            None,
        ),
        (make_scorer(f1_score, response_method="predict", average="macro"), 1),
    ],
)
def test_estimator_report_report_metrics_with_scorer_binary_classification(
    binary_classification_data, scorer, pos_label
):
    """Check that we can pass scikit-learn scorer with different parameters to
    the `report_metrics` method.

    We also check that we can pass `pos_label` whether to the scorer or to the
    `report_metrics` method or consistently to both.
    """
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    result = report.metrics.report_metrics(
        scoring=["accuracy", accuracy_score, scorer],
        scoring_kwargs={"response_method": "predict"},
    )
    assert result.shape == (3, 1)
    np.testing.assert_allclose(
        result.to_numpy(),
        [
            [accuracy_score(y_test, estimator.predict(X_test))],
            [accuracy_score(y_test, estimator.predict(X_test))],
            [f1_score(y_test, estimator.predict(X_test), average="macro", pos_label=1)],
        ],
    )


def test_estimator_report_report_metrics_with_scorer_pos_label_error(
    binary_classification_data,
):
    """Check that we raise an error when pos_label is passed both in the scorer and
    globally conducting to a mismatch."""
    estimator, X_test, y_test = binary_classification_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    f1_scorer = make_scorer(
        f1_score, response_method="predict", average="macro", pos_label=1
    )
    err_msg = re.escape(
        "`pos_label` is passed both in the scorer and to the `report_metrics` method."
    )
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.report_metrics(scoring=[f1_scorer], pos_label=0)


def test_estimator_report_report_metrics_invalid_metric_type(regression_data):
    """Check that we raise the expected error message if an invalid metric is passed."""
    estimator, X_test, y_test = regression_data
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    err_msg = re.escape("Invalid type of metric: <class 'int'> for 1")
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.report_metrics(scoring=[1])


def test_estimator_report_get_X_y_and_data_source_hash_error():
    """Check that we raise the proper error in `get_X_y_and_use_cache`."""
    X, y = make_classification(n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    estimator = LogisticRegression().fit(X_train, y_train)
    report = EstimatorReport(estimator)

    err_msg = re.escape(
        "Invalid data source: unknown. Possible values are: test, train, X_y."
    )
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.log_loss(data_source="unknown")

    for data_source in ("train", "test"):
        err_msg = re.escape(
            f"No {data_source} data (i.e. X_{data_source} and y_{data_source}) were "
            f"provided when creating the report. Please provide the {data_source} "
            "data either when creating the report or by setting data_source to "
            "'X_y' and providing X and y."
        )
        with pytest.raises(ValueError, match=err_msg):
            report.metrics.log_loss(data_source=data_source)

    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    for data_source in ("train", "test"):
        err_msg = f"X and y must be None when data_source is {data_source}."
        with pytest.raises(ValueError, match=err_msg):
            report.metrics.log_loss(data_source=data_source, X=X_test, y=y_test)

    err_msg = "X and y must be provided."
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.log_loss(data_source="X_y")

    # FIXME: once we choose some basic metrics for clustering, then we don't need to
    # use `custom_metric` for them.
    estimator = KMeans().fit(X_train)
    report = EstimatorReport(estimator, X_test=X_test)
    err_msg = "X must be provided."
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.custom_metric(
            rand_score, response_method="predict", data_source="X_y"
        )

    report = EstimatorReport(estimator)
    for data_source in ("train", "test"):
        err_msg = re.escape(
            f"No {data_source} data (i.e. X_{data_source}) were provided when "
            f"creating the report. Please provide the {data_source} data either "
            f"when creating the report or by setting data_source to 'X_y' and "
            f"providing X and y."
        )
        with pytest.raises(ValueError, match=err_msg):
            report.metrics.custom_metric(
                rand_score, response_method="predict", data_source=data_source
            )


@pytest.mark.parametrize("data_source", ("train", "test", "X_y"))
def test_estimator_report_get_X_y_and_data_source_hash(data_source):
    """Check the general behaviour of `get_X_y_and_use_cache`."""
    X, y = make_classification(n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    estimator = LogisticRegression()
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    kwargs = {"X": X_test, "y": y_test} if data_source == "X_y" else {}
    X, y, data_source_hash = report.metrics._get_X_y_and_data_source_hash(
        data_source=data_source, **kwargs
    )

    if data_source == "train":
        assert X is X_train
        assert y is y_train
        assert data_source_hash is None
    elif data_source == "test":
        assert X is X_test
        assert y is y_test
        assert data_source_hash is None
    elif data_source == "X_y":
        assert X is X_test
        assert y is y_test
        assert data_source_hash == joblib.hash((X_test, y_test))


@pytest.mark.parametrize("prefit_estimator", [True, False])
def test_estimator_has_side_effects(prefit_estimator):
    """Re-fitting the estimator outside the EstimatorReport
    should not have an effect on the EstimatorReport's internal estimator."""
    X, y = make_classification(n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    estimator = LogisticRegression()
    if prefit_estimator:
        estimator.fit(X_train, y_train)

    report = EstimatorReport(
        estimator,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    predictions_before = report.estimator_.predict_proba(X_test)
    estimator.fit(X_test, y_test)
    predictions_after = report.estimator_.predict_proba(X_test)
    np.testing.assert_array_equal(predictions_before, predictions_after)


def test_estimator_has_no_deep_copy():
    """Check that we raise a warning if the deep copy failed with a fitted
    estimator."""
    X, y = make_classification(n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    estimator = LogisticRegression()
    # Make it so deepcopy does not work
    estimator.__reduce_ex__ = None
    estimator.__reduce__ = None

    with pytest.warns(UserWarning, match="Deepcopy failed"):
        EstimatorReport(
            estimator,
            fit=False,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )


def test_estimator_report_brier_score_requires_probabilities():
    """Check that the Brier score is not defined for estimator that do not
    implement `predict_proba`.

    Non-regression test for:
    https://github.com/probabl-ai/skore/pull/1471
    """
    estimator = SVC()  # SVC does not implement `predict_proba` with default parameters
    X, y = make_classification(n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    assert not hasattr(report.metrics, "brier_score")
