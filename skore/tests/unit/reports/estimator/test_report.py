import inspect
from copy import deepcopy
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from skore import EstimatorReport


def test_report_can_be_rebuilt_using_parameters(linear_regression_with_test):
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    parameters = {}

    for parameter in inspect.signature(EstimatorReport).parameters:
        assert hasattr(report, parameter), f"The parameter '{parameter}' must be stored"

        parameters[parameter] = getattr(report, parameter)

    EstimatorReport(**parameters)


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
def test_from_unfitted_estimator(fit):
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

    with pytest.raises(AttributeError):
        report.estimator_ = LinearRegression()
    with pytest.raises(AttributeError):
        report.X_train = X_train
    with pytest.raises(AttributeError):
        report.y_train = y_train


@pytest.mark.parametrize("fit", [False, "auto"])
def test_from_fitted_estimator(forest_binary_classification_with_test, fit):
    """Check the general behaviour of passing an already fitted estimator without
    refitting it."""
    estimator, X, y = forest_binary_classification_with_test
    report = EstimatorReport(estimator, fit=fit, X_test=X, y_test=y)

    check_is_fitted(report.estimator_)
    assert isinstance(report.estimator_, RandomForestClassifier)
    assert report.X_train is None
    assert report.y_train is None
    assert report.X_test is X
    assert report.y_test is y

    with pytest.raises(AttributeError):
        report.estimator_ = LinearRegression()
    with pytest.raises(AttributeError):
        report.X_train = X
    with pytest.raises(AttributeError):
        report.y_train = y


def test_from_fitted_pipeline(
    pipeline_binary_classification_with_test,
):
    """Check the general behaviour of passing an already fitted pipeline without
    refitting it.
    """
    estimator, X, y = pipeline_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X, y_test=y)

    check_is_fitted(report.estimator_)
    assert isinstance(report.estimator_, Pipeline)
    assert report.estimator_name_ == estimator[-1].__class__.__name__
    assert report.X_train is None
    assert report.y_train is None
    assert report.X_test is X
    assert report.y_test is y


def test_invalidate_cache_data(forest_binary_classification_with_test):
    """Check that we invalidate the cache when the data is changed."""
    estimator, X_test, y_test = forest_binary_classification_with_test
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
def test_check_support_plot(
    Estimator, X_test, y_test, supported_plot_methods, not_supported_plot_methods
):
    """Check that the available plot methods are correctly registered."""
    estimator = Estimator.fit(X_test, y_test)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    for supported_plot_method in supported_plot_methods:
        assert hasattr(report.metrics, supported_plot_method)

    for not_supported_plot_method in not_supported_plot_methods:
        assert not hasattr(report.metrics, not_supported_plot_method)


@pytest.mark.parametrize(
    "fixture_name, pass_train_data, expected_n_keys",
    [
        ("forest_binary_classification_with_test", True, 10),
        ("svc_binary_classification_with_test", True, 10),
        ("forest_multiclass_classification_with_test", True, 12),
        ("linear_regression_with_test", True, 4),
        ("forest_binary_classification_with_test", False, 5),
        ("svc_binary_classification_with_test", False, 5),
        ("forest_multiclass_classification_with_test", False, 6),
        ("linear_regression_with_test", False, 2),
    ],
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_cache_predictions(
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


def test_pickle(forest_binary_classification_with_test):
    """Check that we can pickle an estimator report.

    In particular, the progress bar from rich are pickable, therefore we trigger
    the progress bar to be able to test that the progress bar is pickable.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    report.cache_predictions()

    with BytesIO() as stream:
        joblib.dump(report, stream)


def test_flat_index(forest_binary_classification_with_test):
    """Check that the index is flattened when `flat_index` is True.

    Since `pos_label` is None, then by default a MultiIndex would be returned.
    Here, we force to have a single-index by passing `flat_index=True`.
    """
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    result = report.metrics.summarize(flat_index=True).frame()
    assert result.shape == (9, 1)
    assert isinstance(result.index, pd.Index)
    assert result.index.tolist() == [
        "accuracy",
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
        "roc_auc",
        "brier_score",
        "fit_time_s",
        "predict_time_s",
    ]

    assert result.columns.tolist() == ["RandomForestClassifier"]


def test_get_predictions():
    """Check the behaviour of the `get_predictions` method.

    We use the binary classification because it uses all the parameters of the
    `get_predictions` method.
    """
    X, y = make_classification(n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator = LogisticRegression()
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    # check the `predict` method
    predictions = report.get_predictions(data_source="test")
    np.testing.assert_allclose(predictions, report.estimator_.predict(X_test))
    predictions = report.get_predictions(data_source="train")
    np.testing.assert_allclose(predictions, report.estimator_.predict(X_train))
    predictions = report.get_predictions(data_source="X_y", X=X_test)
    np.testing.assert_allclose(predictions, report.estimator_.predict(X_test))

    # check the validity of the `predict_proba` method
    predictions = report.get_predictions(
        data_source="test", response_method="predict_proba"
    )
    np.testing.assert_allclose(
        predictions, report.estimator_.predict_proba(X_test)[:, 1]
    )
    predictions = report.get_predictions(
        data_source="train", response_method="predict_proba", pos_label=0
    )
    np.testing.assert_allclose(
        predictions, report.estimator_.predict_proba(X_train)[:, 0]
    )
    predictions = report.get_predictions(
        data_source="X_y", response_method="predict_proba", X=X_test
    )
    np.testing.assert_allclose(
        predictions, report.estimator_.predict_proba(X_test)[:, 1]
    )

    # check the validity of the `decision_function` method
    predictions = report.get_predictions(
        data_source="test", response_method="decision_function"
    )
    np.testing.assert_allclose(predictions, report.estimator_.decision_function(X_test))
    predictions = report.get_predictions(
        data_source="train", response_method="decision_function", pos_label=0
    )
    np.testing.assert_allclose(
        predictions, -report.estimator_.decision_function(X_train)
    )
    predictions = report.get_predictions(
        data_source="X_y", response_method="decision_function", X=X_test
    )
    np.testing.assert_allclose(predictions, report.estimator_.decision_function(X_test))

    # check the behaviour in conjunction of a report `pos_label`
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        pos_label=0,
    )
    predictions = report.get_predictions(
        data_source="train", response_method="predict_proba"
    )
    np.testing.assert_allclose(
        predictions, report.estimator_.predict_proba(X_train)[:, 0]
    )


def test_get_predictions_error():
    """Check that we raise an error when the data source is invalid."""
    X, y = make_classification(n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator = LogisticRegression()
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    with pytest.raises(ValueError, match="Invalid data source"):
        report.get_predictions(data_source="invalid")


def test_clustering():
    """Check that we cannot create a report with a clustering model."""
    with pytest.raises(
        ValueError,
        match="Clustering models are not supported yet. Please use a "
        "classification or regression model instead.",
    ):
        EstimatorReport(KMeans())
