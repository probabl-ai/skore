import re
from numbers import Real

import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from skore import EstimatorReport


def deep_contain(value, test_value):
    if value == test_value:
        return True
    elif isinstance(value, tuple):
        return any(deep_contain(item, test_value) for item in value)
    else:
        return False


def test_interaction_cache_metrics(
    linear_regression_multioutput_with_test,
):
    """Check that the cache take into account the 'kwargs' of a metric."""
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    # The underlying metrics will call `_compute_metric_scores` that take some arbitrary
    # kwargs apart from `pos_label`. Let's pass an arbitrary kwarg and make sure it is
    # part of the cache.
    multioutput = "raw_values"
    result_r2_raw_values = report.metrics.r2(multioutput=multioutput)
    multioutput_in_cache_key = any(
        deep_contain(cached_key, multioutput) for cached_key in report._cache
    )
    assert multioutput_in_cache_key, (
        f"The value {multioutput} should be stored in one of the cache keys"
    )
    assert len(result_r2_raw_values) == 2

    multioutput = "uniform_average"
    result_r2_uniform_average = report.metrics.r2(multioutput=multioutput)
    multioutput_in_cache_key = any(
        deep_contain(cached_key, multioutput) for cached_key in report._cache
    )
    assert multioutput_in_cache_key, (
        f"The value {multioutput} should be stored in one of the cache keys"
    )
    assert isinstance(result_r2_uniform_average, float)


def test_custom_metric(linear_regression_with_test):
    """Check the behaviour of the `custom_metric` computation in the report."""
    estimator, X_test, y_test = linear_regression_with_test
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
    threshold_in_cache_key = any(
        deep_contain(cached_key, threshold) for cached_key in report._cache
    )
    assert threshold_in_cache_key, (
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
    threshold_in_cache_key = any(
        deep_contain(cached_key, threshold) for cached_key in report._cache
    )
    assert threshold_in_cache_key, (
        f"The value {threshold} should be stored in one of the cache keys"
    )

    assert isinstance(result, float)
    assert result == pytest.approx(
        custom_metric(y_test, estimator.predict(X_test), threshold)
    )


def test_custom_metric_scorer(linear_regression_with_test):
    """Check the behaviour of the `custom_metric` computation in the report when the
    custom metric is a sklearn Scorer."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    def custom_metric(y_true, y_pred, threshold=0.5):
        residuals = y_true - y_pred
        return np.mean(np.where(residuals < threshold, residuals, 1))

    custom_score = make_scorer(custom_metric, greater_is_better=False)

    threshold = 1
    result = report.metrics.custom_metric(
        metric_function=custom_score,
        response_method="predict",
        threshold=threshold,
    )
    threshold_in_cache_key = any(
        deep_contain(cached_key, threshold) for cached_key in report._cache
    )
    assert threshold_in_cache_key, (
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
    threshold_in_cache_key = any(
        deep_contain(cached_key, threshold) for cached_key in report._cache
    )
    assert threshold_in_cache_key, (
        f"The value {threshold} should be stored in one of the cache keys"
    )

    assert isinstance(result, float)
    assert result == pytest.approx(
        custom_metric(y_test, estimator.predict(X_test), threshold)
    )


def test_custom_function_kwargs_numpy_array(
    linear_regression_with_test,
):
    """Check that we are able to store a hash of a numpy array in the cache when they
    are passed as kwargs.
    """
    estimator, X_test, y_test = linear_regression_with_test
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
    hash_weights_in_cache_key = any(
        deep_contain(cached_key, hash_weights) for cached_key in report._cache
    )
    assert hash_weights_in_cache_key, (
        "The hash of the weights should be stored in one of the cache keys"
    )

    assert isinstance(result, float)
    assert result == pytest.approx(
        custom_metric(y_test, estimator.predict(X_test), weights)
    )


def test_custom_metric_compatible_estimator(
    forest_binary_classification_with_test,
):
    """Check that the estimator report still works if an estimator has a compatible
    scikit-learn API.
    """
    _, X_test, y_test = forest_binary_classification_with_test

    class CompatibleEstimator(BaseEstimator):
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


def test_get_X_y_and_data_source_hash_error():
    """Check that we raise the proper error in `_get_X_y_and_data_source_hash`."""
    X, y = make_classification(n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    estimator = LogisticRegression().fit(X_train, y_train)
    report = EstimatorReport(estimator)

    err_msg = re.escape(
        "Invalid data source: unknown. Possible values are: test, train, X_y."
    )
    with pytest.raises(ValueError, match=err_msg):
        report.metrics._get_X_y_and_data_source_hash(data_source="unknown")

    for data_source in ("train", "test"):
        err_msg = re.escape(
            f"No {data_source} data (i.e. X_{data_source} and y_{data_source}) were "
            f"provided when creating the report. Please provide the {data_source} "
            "data either when creating the report or by setting data_source to "
            "'X_y' and providing X and y."
        )
        with pytest.raises(ValueError, match=err_msg):
            report.metrics._get_X_y_and_data_source_hash(data_source=data_source)

    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    for data_source in ("train", "test"):
        err_msg = f"X and y must be None when data_source is {data_source}."
        with pytest.raises(ValueError, match=err_msg):
            report.metrics._get_X_y_and_data_source_hash(
                data_source=data_source, X=X_test, y=y_test
            )

    err_msg = "X and y must be provided."
    with pytest.raises(ValueError, match=err_msg):
        report.metrics._get_X_y_and_data_source_hash(data_source="X_y")


@pytest.mark.parametrize("data_source", ("train", "test", "X_y"))
def test_get_X_y_and_data_source_hash(data_source):
    """Check the general behaviour of `_get_X_y_and_data_source_hash`."""
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
def test_has_side_effects(prefit_estimator):
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


@pytest.mark.parametrize("metric", ["brier_score", "log_loss"])
def test_brier_log_loss_requires_probabilities(metric):
    """Check that the Brier score and the Log loss is not defined for estimator
    that do not implement `predict_proba`.

    Non-regression test for:
    https://github.com/probabl-ai/skore/pull/1471
    https://github.com/probabl-ai/skore/issues/1736
    """
    estimator = SVC()  # SVC does not implement `predict_proba` with default parameters
    X, y = make_classification(n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    assert not hasattr(report.metrics, metric)


def test_brier_score_requires_binary_classification():
    """Check that the Brier score is not defined for estimator that do not
    implement `predict_proba` and that are not binary-classification.

    Non-regression test for:
    https://github.com/probabl-ai/skore/issues/1540
    """
    estimator = LogisticRegression()
    X, y = make_classification(n_classes=3, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    assert not hasattr(report.metrics, "brier_score")


def test_average_return_float(forest_binary_classification_with_test):
    """Check that we expect a float value when computing a metric with averaging.

    Non-regression test for:
    https://github.com/probabl-ai/skore/issues/1501
    """
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    for metric_name in ("precision", "recall", "roc_auc"):
        result = getattr(report.metrics, metric_name)(average="macro")
        assert isinstance(result, float)


@pytest.mark.parametrize(
    "metric, metric_fn", [("precision", precision_score), ("recall", recall_score)]
)
def test_precision_recall_pos_label_overwrite(metric, metric_fn):
    """Check that `pos_label` can be overwritten."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    classifier = LogisticRegression().fit(X, y)

    report = EstimatorReport(classifier, X_test=X, y_test=y)
    result = getattr(report.metrics, metric)()
    assert result.keys() == {"A", "B"}

    report = EstimatorReport(classifier, X_test=X, y_test=y, pos_label="B")
    assert getattr(report.metrics, metric)(pos_label="B") == pytest.approx(
        metric_fn(y, classifier.predict(X), pos_label="B")
    )
    assert getattr(report.metrics, metric)(pos_label="A") == pytest.approx(
        metric_fn(y, classifier.predict(X), pos_label="A")
    )


def test_roc_multiclass_requires_predict_proba(
    svc_multiclass_classification_with_test, svc_binary_classification_with_test
):
    """Check that the ROC AUC metric is not exposed with multiclass data if the
    estimator does not expose `predict_proba`.

    Non-regression test for:
    https://github.com/probabl-ai/skore/issues/1873
    """
    classifier, X_test, y_test = svc_multiclass_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)
    with pytest.raises(AttributeError):
        report.metrics.roc_auc()

    classifier, X_test, y_test = svc_binary_classification_with_test
    report = EstimatorReport(classifier, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, "roc_auc")
    report.metrics.roc_auc()
