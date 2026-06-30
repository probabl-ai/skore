import warnings

import numpy as np
import pandas as pd
import pytest
import skrub
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
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


def test_interaction_cache_metrics(linear_regression_multioutput_with_test):
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


def test_r2_returns_float(linear_regression_with_test):
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    assert isinstance(report.metrics.r2(), float)


# report.metrics.score


def test_score_matches_sklearn_score(logistic_binary_classification_with_train_test):
    """For a plain sklearn estimator, ``score`` returns ``estimator.score(X, y)``."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    assert report.metrics.score() == report.estimator_.score(X_test, y_test)


def skrub_report(*, with_scoring):
    X, y = make_classification(random_state=0)

    # String-labeled y avoids the int/str clash in MetricsSummaryDisplay's label
    # column when ``Score`` emits a dict keyed by scorer name.
    y = np.where(y == 1, "pos", "neg")

    data_op = skrub.X(X).skb.apply(DummyClassifier(), y=skrub.y(y))
    if with_scoring:
        data_op = data_op.skb.with_scoring("accuracy").skb.with_scoring(
            "accuracy", name="weighted_accuracy"
        )

    learner = data_op.skb.make_learner()
    split = data_op.skb.train_test_split()
    return EstimatorReport(learner, train_data=split["train"], test_data=split["test"])


@pytest.mark.parametrize("with_scoring", [False, True])
def test_score_skrub_learner(with_scoring):
    """``score`` on a report containing a SkrubLearner returns the same result
    as ``score`` on the learner directly.

    Non-regression test: previously ``SkrubLearner.score`` was called as
    ``score(X, y)`` but it expects an environment dict.
    """
    report = skrub_report(with_scoring=with_scoring)

    assert report.metrics.score() == report.estimator_.score(
        {"_skrub_X": report.X_test, "_skrub_y": report.y_test}
    )


def test_score_skrub_learner_with_extra_env_vars():
    """``score`` works when the DataOp env has variables beyond X and y."""
    df = pd.DataFrame({"feat": np.arange(20, dtype=float), "target": ["a", "b"] * 10})
    data = skrub.var("df", df)
    weight = skrub.var("weight", 0.5)

    X = data[["feat"]].skb.mark_as_X()
    weighted_X = X * weight  # brings in `weight` after marking the X
    y = data["target"].skb.mark_as_y()

    data_op = weighted_X.skb.apply(DummyClassifier(), y=y)
    learner = data_op.skb.make_learner()
    split = data_op.skb.train_test_split()
    report = EstimatorReport(
        learner, train_data=split["train"], test_data=split["test"]
    )

    assert isinstance(report.metrics.score(), float)


# report.metrics.get


def test_get(binary_classification_report):
    """``get`` works."""
    report = binary_classification_report

    assert report.metrics.get("precision") == 1

    with pytest.raises(KeyError):
        report.metrics.get("non-existing metric")


def test_get_custom(binary_classification_report):
    """``get`` works for custom metrics."""
    report = binary_classification_report

    with pytest.raises(KeyError):
        report.metrics.get("hello")

    report.metrics.add(lambda estimator, X, y: 1, name="hello")

    assert report.metrics.get("hello") == 1
