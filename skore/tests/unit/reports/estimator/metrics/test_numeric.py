import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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

    result_r2_raw_values = report.metrics.r2()
    assert len(result_r2_raw_values) == 2

    multioutput_in_cache_key = any(
        deep_contain(cached_key, "raw_values") for cached_key in report._cache
    )
    assert multioutput_in_cache_key, (
        "The value 'raw_values' should be stored in one of the cache keys"
    )


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
    """Check that we expect a float value when computing a metric with pos_label.

    Non-regression test for:
    https://github.com/probabl-ai/skore/issues/1501
    """
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test, pos_label=1)

    for metric_name in ("precision", "recall", "roc_auc"):
        result = getattr(report.metrics, metric_name)()
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
    assert getattr(report.metrics, metric)() == pytest.approx(
        metric_fn(y, classifier.predict(X), pos_label="B")
    )

    report = EstimatorReport(classifier, X_test=X, y_test=y, pos_label="A")
    assert getattr(report.metrics, metric)() == pytest.approx(
        metric_fn(y, classifier.predict(X), pos_label="A")
    )


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_precision_recall_macro_average_ignores_pos_label(metric):
    """Check that macro-averaged precision/recall does not forward pos_label."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    y = np.array(["negative", "positive"], dtype=object)[y]
    classifier = LogisticRegression(max_iter=1000).fit(X, y)
    report = EstimatorReport(classifier, X_test=X, y_test=y, pos_label="positive")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = getattr(report.metrics, metric)(average="macro")

    assert isinstance(result, float)

    result = getattr(report.metrics, metric)(average="binary")
    assert isinstance(result, float)


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
