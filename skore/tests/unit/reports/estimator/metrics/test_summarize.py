"""Tests for EstimatorReport.metrics.summarize().

Organised by metric input type, then corner cases:

- Default metrics — by ML task variant
- Metric strings — skore built-in registry names
- pos_label
- Cache and data_source
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from skore import EstimatorReport, MetricsSummaryDisplay
from skore._utils._testing import check_cache_changed, check_cache_unchanged


def check_display_structure(
    display,
    *,
    expected_metrics,
    expected_estimator_name=None,
    expected_data_source="test",
    expected_greater_is_better=None,
    expected_average=None,
):
    """Check the full structure of a MetricsSummaryDisplay.data DataFrame."""
    assert isinstance(display.data, pd.DataFrame)
    data = display.data

    assert set(data.columns) == {
        "metric",
        "estimator_name",
        "data_source",
        "label",
        "average",
        "output",
        "score",
        "greater_is_better",
    }
    assert pd.api.types.is_numeric_dtype(data["score"])
    assert set(data["metric"]) == expected_metrics
    assert set(data["estimator_name"]) == {expected_estimator_name}
    assert set(data["data_source"]) == {expected_data_source}
    if expected_average is None:
        assert data["average"].isna().all()
    else:
        assert set(data["average"]) == expected_average
    if expected_greater_is_better is None:
        expected_greater_is_better = {True, False}
    assert set(data["greater_is_better"]) == expected_greater_is_better


# Default metrics


@pytest.mark.parametrize("metric", [None, [], ()])
def test_default(forest_binary_classification_with_test, metric):
    """If no metric is passed then use the ML task defaults."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric=metric)

    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Precision",
            "Recall",
            "ROC AUC",
            "Log loss",
            "Brier score",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="RandomForestClassifier",
    )


def test_default_binary_classification_svc(svc_binary_classification_with_test):
    """If a model has no predict_proba then it will have no Log loss or Brier score."""
    estimator, X_test, y_test = svc_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test, pos_label=1)
    display = report.metrics.summarize()

    assert isinstance(display.data, pd.DataFrame)
    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Precision",
            "Recall",
            "ROC AUC",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="SVC",
    )


def test_default_multiclass_classification_forest(
    forest_multiclass_classification_with_test,
):
    """Multiclass classification with RandomForestClassifier."""
    estimator, X_test, y_test = forest_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Log loss",
            "Precision",
            "Recall",
            "ROC AUC",
            "Predict time (s)",
            "Fit time (s)",
        },
        expected_estimator_name="RandomForestClassifier",
    )

    assert display.data["output"].isna().all()
    data = display.data.set_index("metric")
    assert len(data.loc["Precision"]) == 3
    assert len(data.loc["Recall"]) == 3
    assert set(data.loc["Precision", "label"]) == {0, 1, 2}


def test_default_multiclass_classification_svc(svc_multiclass_classification_with_test):
    """Multiclass classification with SVC (no predict_proba)."""
    estimator, X_test, y_test = svc_multiclass_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Precision",
            "Recall",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="SVC",
    )

    assert display.data["output"].isna().all()
    data = display.data.set_index("metric")
    assert len(data.loc["Precision"]) == 3
    assert len(data.loc["Recall"]) == 3
    assert set(data.loc["Precision", "label"]) == {0, 1, 2}


def test_default_regression(linear_regression_with_test):
    """Regression with LinearRegression."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={"R²", "RMSE", "Fit time (s)", "Predict time (s)"},
        expected_estimator_name="LinearRegression",
    )

    assert display.data["label"].isna().all()
    assert display.data["output"].isna().all()


def test_default_multioutput_regression(linear_regression_multioutput_with_test):
    """Multioutput regression with LinearRegression."""
    estimator, X_test, y_test = linear_regression_multioutput_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={"R²", "RMSE", "Fit time (s)", "Predict time (s)"},
        expected_estimator_name="LinearRegression",
    )

    assert display.data["label"].isna().all()
    data = display.data.set_index("metric")
    assert len(data.loc["R²", "output"]) == 2
    assert len(data.loc["RMSE", "output"]) == 2
    assert set(data.loc["R²", "output"]) == {0, 1}


def test_default_without_predict_proba(custom_classifier_no_predict_proba_with_test):
    """Default metrics skip roc_auc, log_loss, and brier_score without predict_proba."""
    estimator, X_test, y_test = custom_classifier_no_predict_proba_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Precision",
            "Recall",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="CustomClassifierPredictOnly",
    )


# Metric strings


def test_string_plain(linear_regression_with_test):
    """A list of skore built-in metric strings resolves to correct display names."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize(metric=["r2", "rmse"])

    check_display_structure(
        display,
        expected_metrics={"R²", "RMSE"},
        expected_estimator_name="LinearRegression",
    )


# pos_label


def test_pos_label(forest_binary_classification_with_test):
    """pos_label collapses per-class metrics to a single row."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test, pos_label=1)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "Accuracy",
            "Precision",
            "Recall",
            "ROC AUC",
            "Log loss",
            "Brier score",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="RandomForestClassifier",
    )

    assert len(display.data[display.data["metric"] == "Precision"]) == 1
    assert len(display.data[display.data["metric"] == "Recall"]) == 1
    assert display.data["label"].isna().all()
    assert display.data["output"].isna().all()


def test_pos_label_strings(forest_binary_classification_with_test):
    """Binary classification with string labels."""
    estimator, X_test, y_test = forest_binary_classification_with_test

    target_names = np.array(["neg", "pos"], dtype=object)
    y_test = target_names[y_test]

    estimator = clone(estimator).fit(X_test, y_test)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize()
    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["metric"]) == {
        "Accuracy",
        "Precision",
        "Recall",
        "ROC AUC",
        "Log loss",
        "Brier score",
        "Fit time (s)",
        "Predict time (s)",
    }

    labels = display.data.set_index("metric").loc["Precision", "label"]
    assert set(labels) == {"neg", "pos"}


def test_pos_label_bool(forest_binary_classification_with_test):
    """Binary classification with boolean labels."""
    estimator, X_test, y_test = forest_binary_classification_with_test

    target_names = np.array([False, True], dtype=bool)
    y_test = target_names[y_test]

    estimator = clone(estimator).fit(X_test, y_test)
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    display = report.metrics.summarize()
    assert isinstance(display.data, pd.DataFrame)
    assert set(display.data["metric"]) == {
        "Accuracy",
        "Precision",
        "Recall",
        "ROC AUC",
        "Log loss",
        "Brier score",
        "Fit time (s)",
        "Predict time (s)",
    }

    labels = display.data.set_index("metric").loc["Precision", "label"]
    assert any(label is np.False_ for label in labels)
    assert any(label is np.True_ for label in labels)


@pytest.mark.parametrize(
    "metric, metric_fn", [("precision", precision_score), ("recall", recall_score)]
)
def test_pos_label_overwrite(metric, metric_fn):
    """pos_label can be set when creating the report."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    classifier = LogisticRegression().fit(X, y)

    # Without pos_label - should have multiple rows (one per class)
    report = EstimatorReport(classifier, X_test=X, y_test=y)
    display = report.metrics.summarize(metric=metric)
    assert isinstance(display.data, pd.DataFrame)
    assert len(display.data) == 2
    assert set(display.data["label"]) == {"A", "B"}

    # With pos_label="B" - should have single row
    report = EstimatorReport(classifier, X_test=X, y_test=y, pos_label="B")
    display = report.metrics.summarize(metric=metric)
    assert len(display.data) == 1
    score_B = display.data["score"].values[0]
    assert score_B == pytest.approx(metric_fn(y, classifier.predict(X), pos_label="B"))

    # With pos_label="A" - should have single row
    report = EstimatorReport(classifier, X_test=X, y_test=y, pos_label="A")
    display = report.metrics.summarize(metric=metric)
    assert len(display.data) == 1
    score_A = display.data["score"].values[0]
    assert score_A == pytest.approx(metric_fn(y, classifier.predict(X), pos_label="A"))


# Cache and data_source


def test_cache(forest_binary_classification_with_test):
    """summarize() results are cached; second call returns the same data."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

    with check_cache_changed(report._cache):
        result = report.metrics.summarize()
    assert isinstance(result, MetricsSummaryDisplay)

    with check_cache_unchanged(report._cache):
        result_from_cache = report.metrics.summarize()
    assert_frame_equal(result.data, result_from_cache.data)


def test_data_source_both(forest_binary_classification_data):
    """data_source='both' concatenates train and test results."""
    estimator, X, y = forest_binary_classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display_train = report.metrics.summarize(data_source="train")
    display_test = report.metrics.summarize(data_source="test")
    display_both = report.metrics.summarize(data_source="both")

    assert set(display_both.data["data_source"]) == {"train", "test"}

    train_data = display_both.data[display_both.data["data_source"] == "train"]
    assert_array_equal(train_data["score"], display_train.data["score"])

    test_data = display_both.data[display_both.data["data_source"] == "test"]
    assert_array_equal(test_data["score"], display_test.data["score"])
