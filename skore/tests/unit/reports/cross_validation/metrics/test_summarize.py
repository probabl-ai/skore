"""Tests for CrossValidationReport.metrics.summarize() method.

These tests focus on testing the data aggregation logic of summarize()
without depending on MetricsSummaryDisplay.frame().
"""

import numpy as np
import pandas as pd
import pytest

from skore import CrossValidationReport, MetricsSummaryDisplay


def check_display_structure(
    display,
    *,
    expected_metrics,
    expected_estimator_name=None,
    expected_data_source="test",
    expected_favorability=None,
):
    """
    Helper function to check the structure of a MetricsSummaryDisplay.data DataFrame.

    Parameters
    ----------
    display : MetricsSummaryDisplay
        The display object to check.
    expected_metrics : set, optional
        Expected set of metric names.
    expected_estimator_name : str, optional
        Expected estimator name.
    expected_data_source : str, default="test"
        Expected data source value.
    expected_favorability : set, optional
        Expected set of favorability indicators.
    """
    assert isinstance(display, MetricsSummaryDisplay)
    assert isinstance(display.data, pd.DataFrame)
    data = display.data

    assert set(data.columns) == {
        "split",
        "metric",
        "estimator_name",
        "data_source",
        "label",
        "average",
        "output",
        "score",
        "favorability",
    }
    assert set(data["metric"]) == expected_metrics
    assert set(data["estimator_name"]) == {expected_estimator_name}
    assert set(data["data_source"]) == {expected_data_source}
    assert set(data["split"]) == {0, 1}
    assert pd.api.types.is_numeric_dtype(data["score"])
    assert pd.api.types.is_integer_dtype(data["split"])
    if expected_favorability is None:
        assert set(data["favorability"]) == {"(↗︎)", "(↘︎)"}
    else:
        assert set(data["favorability"]) == expected_favorability


# Tests for the happy path, with different ML tasks


def test_binary_classification_forest(forest_binary_classification_data):
    """
    Check the behaviour of summarize() with binary classification using
    RandomForestClassifier.
    """
    estimator, X, y = forest_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
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

    data = display.data.set_index(["split", "metric"]).sort_index()
    assert len(data.loc[(0, "Precision")]) == 2
    assert len(data.loc[(0, "Recall")]) == 2

    assert set(display.data.set_index("metric").loc["Precision", "label"]) == {0, 1}
    assert display.data["output"].isna().all()


def test_binary_classification_svc(svc_binary_classification_data):
    """
    Check the behaviour of summarize() with binary classification using SVC
    (no predict_proba).
    """
    estimator, X, y = svc_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2, pos_label=1)
    display = report.metrics.summarize()

    # No Brier score for SVC
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


def test_multiclass_classification_forest(forest_multiclass_classification_data):
    """
    Check the behaviour of summarize() with multiclass classification using
    RandomForestClassifier.
    """
    estimator, X, y = forest_multiclass_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
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

    data = display.data.set_index(["split", "metric"]).sort_index()
    # 3 classes
    assert (
        set(data.loc[(0, "Precision"), "label"])
        == set(data.loc[(0, "Recall"), "label"])
        == set(data.loc[(0, "ROC AUC"), "label"])
        == {0, 1, 2}
    )


def test_multiclass_classification_svc(svc_multiclass_classification_data):
    """Check the behaviour of summarize() with multiclass classification using SVC."""
    estimator, X, y = svc_multiclass_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
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

    data = display.data.set_index(["split", "metric"]).sort_index()
    assert len(data.loc[(0, "Precision")]) == 3
    assert len(data.loc[(0, "Recall")]) == 3


def test_regression(linear_regression_data):
    """Check the behaviour of summarize() with regression."""
    estimator, X, y = linear_regression_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "R²",
            "RMSE",
            "MAE",
            "MAPE",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="LinearRegression",
    )

    assert display.data["label"].isna().all()
    assert display.data["output"].isna().all()


def test_multioutput_regression(linear_regression_multioutput_data):
    """Check the behaviour of summarize() with multioutput regression."""
    estimator, X, y = linear_regression_multioutput_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "R²",
            "RMSE",
            "MAE",
            "MAPE",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="LinearRegression",
    )

    assert display.data["label"].isna().all()

    data = display.data.set_index(["split", "metric"]).sort_index()
    assert len(data.loc[(0, "R²")]) == 2
    assert set(data.loc[(0, "R²"), "output"]) == {0, 1}


def test_without_predict_proba(custom_classifier_no_predict_proba_data):
    """Default metrics skip roc_auc, log_loss, and brier_score without predict_proba."""
    estimator, X, y = custom_classifier_no_predict_proba_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    display = report.metrics.summarize()

    check_display_structure(
        display,
        expected_metrics={
            "Precision",
            "Accuracy",
            "Recall",
            "Fit time (s)",
            "Predict time (s)",
        },
        expected_estimator_name="CustomClassifierPredictOnly",
    )


# Tests about default metric behavior


def test_default_multioutput_regression(linear_regression_multioutput_data):
    """Default summarize() produces per-output rows for multioutput regression."""
    estimator, X, y = linear_regression_multioutput_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)

    display = report.metrics.summarize()
    assert isinstance(display, MetricsSummaryDisplay)

    # Each metric should have 2 outputs per split
    assert (
        len(display.data.set_index(["split", "metric"]).sort_index().loc[(0, "R²")])
        == 2
    )


def test_default_multiclass_classification(forest_multiclass_classification_data):
    """Default summarize() produces per-class rows for multiclass classification."""
    estimator, X, y = forest_multiclass_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=2)

    display = report.metrics.summarize()
    assert isinstance(display, MetricsSummaryDisplay)

    assert (
        len(
            display.data.set_index(["split", "metric"])
            .sort_index()
            .loc[(0, "Precision")]
        )
        == 3
    )


# Tests about passing `pos_label`


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_pos_label_overwrite(metric, logistic_binary_classification_data):
    """Check that `pos_label` can be overwritten in `summarize`."""
    classifier, X, y = logistic_binary_classification_data
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]

    # Map internal names to display names
    metric_display_name = {"precision": "Precision", "recall": "Recall"}[metric]

    # Without pos_label
    report = CrossValidationReport(classifier, X=X, y=y, splitter=2)
    display = report.metrics.summarize(metric=metric)

    data = display.data.set_index(["split", "metric"]).sort_index()
    assert data.loc[(0, metric_display_name), "label"].to_list() == ["A", "B"]

    # With pos_label
    report = CrossValidationReport(classifier, X=X, y=y, splitter=2, pos_label="A")
    display = report.metrics.summarize(metric=metric)

    assert len(display.data) == 2  # One line per split
    assert display.data["label"].isna().all()
