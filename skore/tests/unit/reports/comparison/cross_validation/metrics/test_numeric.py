"""Tests of metrics available in `ComparisonReport.metrics`."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from skore import ComparisonReport, CrossValidationReport

expected_columns = pd.MultiIndex.from_tuples(
    [
        ("mean", "DummyClassifier_1"),
        ("mean", "DummyClassifier_2"),
        ("std", "DummyClassifier_1"),
        ("std", "DummyClassifier_2"),
    ],
    names=[None, "Estimator"],
)


@pytest.fixture
def case_timings_no_predictions(
    comparison_cross_validation_reports_binary_classification,
):
    expected_index = pd.Index(["Fit time (s)"], name="Metric")
    return (
        comparison_cross_validation_reports_binary_classification,
        "timings",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_timings_with_predictions(
    comparison_cross_validation_reports_binary_classification,
):
    expected_index = pd.Index(
        ["Fit time (s)", "Predict time test (s)", "Predict time train (s)"],
        name="Metric",
    )

    report = comparison_cross_validation_reports_binary_classification
    report.cache_predictions()
    return (
        report,
        "timings",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_accuracy(comparison_cross_validation_reports_binary_classification):
    expected_index = pd.Index(["Accuracy"], name="Metric")
    return (
        comparison_cross_validation_reports_binary_classification,
        "accuracy",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_precision(comparison_cross_validation_reports_binary_classification):
    expected_index = pd.MultiIndex.from_tuples(
        [("Precision", 0), ("Precision", 1)], names=["Metric", "Label / Average"]
    )
    return (
        comparison_cross_validation_reports_binary_classification,
        "precision",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_recall(comparison_cross_validation_reports_binary_classification):
    expected_index = pd.MultiIndex.from_tuples(
        [("Recall", 0), ("Recall", 1)], names=["Metric", "Label / Average"]
    )
    return (
        comparison_cross_validation_reports_binary_classification,
        "recall",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_brier_score(comparison_cross_validation_reports_binary_classification):
    expected_index = pd.Index(["Brier score"], name="Metric")
    return (
        comparison_cross_validation_reports_binary_classification,
        "brier_score",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_roc_auc(comparison_cross_validation_reports_binary_classification):
    expected_index = pd.Index(["ROC AUC"], name="Metric")
    return (
        comparison_cross_validation_reports_binary_classification,
        "roc_auc",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_log_loss(comparison_cross_validation_reports_binary_classification):
    expected_index = pd.Index(["Log loss"], name="Metric")
    return (
        comparison_cross_validation_reports_binary_classification,
        "log_loss",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_r2(comparison_cross_validation_reports_regression):
    expected_index = pd.Index(["R²"], name="Metric")
    expected_columns = pd.MultiIndex.from_tuples(
        [
            ("mean", "DummyRegressor_1"),
            ("mean", "DummyRegressor_2"),
            ("std", "DummyRegressor_1"),
            ("std", "DummyRegressor_2"),
        ],
        names=[None, "Estimator"],
    )
    return (
        comparison_cross_validation_reports_regression,
        "r2",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case_rmse(comparison_cross_validation_reports_regression):
    expected_index = pd.Index(["RMSE"], name="Metric")
    expected_columns = pd.MultiIndex.from_tuples(
        [
            ("mean", "DummyRegressor_1"),
            ("mean", "DummyRegressor_2"),
            ("std", "DummyRegressor_1"),
            ("std", "DummyRegressor_2"),
        ],
        names=[None, "Estimator"],
    )
    return (
        comparison_cross_validation_reports_regression,
        "rmse",
        expected_index,
        expected_columns,
    )


@pytest.fixture
def case(request):
    """Fixture to handle indirect parametrization of case fixtures."""
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "case",
    [
        "case_timings_no_predictions",
        "case_timings_with_predictions",
        "case_accuracy",
        "case_precision",
        "case_recall",
        "case_brier_score",
        "case_roc_auc",
        "case_log_loss",
        "case_r2",
        "case_rmse",
    ],
    indirect=True,
)
def test_metrics(case):
    report, scoring, expected_index, expected_columns = case

    result = getattr(report.metrics, scoring)()
    assert_index_equal(result.index, expected_index)
    assert_index_equal(result.columns, expected_columns)


def test_custom_metric(case_accuracy):
    report, scoring, expected_index, expected_columns = case_accuracy

    result = report.metrics.custom_metric(
        metric_function=accuracy_score,
        response_method="predict",
    )
    result = getattr(report.metrics, scoring)()
    assert_index_equal(result.index, expected_index)
    assert_index_equal(result.columns, expected_columns)


@pytest.mark.parametrize(
    "case",
    [
        "case_timings_no_predictions",
        "case_timings_with_predictions",
        "case_accuracy",
        "case_precision",
        "case_recall",
        "case_brier_score",
        "case_roc_auc",
        "case_log_loss",
        "case_r2",
        "case_rmse",
    ],
    indirect=True,
)
def test_metrics_aggregate(case):
    """`aggregate` argument should be taken into account."""
    report, scoring, expected_index, _ = case

    model = "DummyRegressor" if scoring in ("r2", "rmse") else "DummyClassifier"
    expected_columns = pd.MultiIndex.from_tuples(
        [("mean", f"{model}_1"), ("mean", f"{model}_2")], names=[None, "Estimator"]
    )

    result = getattr(report.metrics, scoring)(aggregate=["mean"])
    assert_index_equal(result.index, expected_index)
    assert_index_equal(result.columns, expected_columns)


def test_metrics_X_y(case_accuracy):
    report, _, expected_index, expected_columns = case_accuracy
    X, y = make_classification(class_sep=0.1, random_state=42)
    result = report.metrics.accuracy(data_source="X_y", X=X, y=y)
    assert_index_equal(result.index, expected_index)
    assert_index_equal(result.columns, expected_columns)


@pytest.mark.parametrize("metric", ["roc", "precision_recall"])
def test_binary_classification_pos_label(pyplot, metric):
    """Check the behaviour of the display methods when `pos_label` needs to be set."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    report_1 = CrossValidationReport(LogisticRegression(C=1), X, y)
    report_2 = CrossValidationReport(LogisticRegression(C=2), X, y)
    report = ComparisonReport([report_1, report_2])
    with pytest.raises(ValueError, match="pos_label is not specified"):
        getattr(report.metrics, metric)()

    report_1 = CrossValidationReport(LogisticRegression(C=1), X, y, pos_label="A")
    report_2 = CrossValidationReport(LogisticRegression(C=2), X, y, pos_label="A")
    report = ComparisonReport([report_1, report_2])
    display = getattr(report.metrics, metric)()
    display.plot()
    assert "Positive label: A" in display.ax_.get_xlabel()

    display = getattr(report.metrics, metric)(pos_label="B")
    display.plot()
    assert "Positive label: B" in display.ax_.get_xlabel()


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_pos_label_default(metric):
    """Check the default behaviour of `pos_label` in `summarize`."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]

    report_1 = CrossValidationReport(LogisticRegression(), X, y)
    report_2 = CrossValidationReport(LogisticRegression(), X, y)
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result_both_labels = report.metrics.summarize(metric=metric).frame().reset_index()
    assert result_both_labels["Label / Average"].to_list() == ["A", "B"]
    result_both_labels = result_both_labels.set_index(["Metric", "Label / Average"])


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_pos_label_overwrite(metric):
    """Check that `pos_label` can be overwritten in `summarize`"""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]

    report_1 = CrossValidationReport(LogisticRegression(), X, y, pos_label="B")
    report_2 = CrossValidationReport(LogisticRegression(), X, y, pos_label="B")
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})

    result_both_labels = report.metrics.summarize(metric=metric, pos_label=None).frame()

    result = report.metrics.summarize(metric=metric).frame().reset_index()
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    for report_name in report.reports_:
        assert (
            result.loc[metric.capitalize(), ("mean", report_name)]
            == result_both_labels.loc[(metric.capitalize(), "B"), ("mean", report_name)]
        )

    result = (
        report.metrics.summarize(metric=metric, pos_label="A").frame().reset_index()
    )
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    for report_name in report.reports_:
        assert (
            result.loc[metric.capitalize(), ("mean", report_name)]
            == result_both_labels.loc[(metric.capitalize(), "A"), ("mean", report_name)]
        )


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_precision_recall_pos_label_default(metric):
    """Check the default behaviour of `pos_label` in `summarize`."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    report_1 = CrossValidationReport(LogisticRegression(), X, y)
    report_2 = CrossValidationReport(LogisticRegression(), X, y)
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})
    result_both_labels = getattr(report.metrics, metric)().reset_index()
    assert result_both_labels["Label / Average"].to_list() == ["A", "B"]


@pytest.mark.parametrize("metric", ["precision", "recall"])
def test_precision_recall_pos_label_overwrite(metric):
    """Check that `pos_label` can be overwritten in `summarize`."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    report_1 = CrossValidationReport(LogisticRegression(), X, y)
    report_2 = CrossValidationReport(LogisticRegression(), X, y)
    report = ComparisonReport({"report_1": report_1, "report_2": report_2})

    result_both_labels = getattr(report.metrics, metric)(pos_label=None)

    result = getattr(report.metrics, metric)(pos_label="B").reset_index()
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    for report_name in report.reports_:
        assert (
            result.loc[metric.capitalize(), ("mean", report_name)]
            == result_both_labels.loc[(metric.capitalize(), "B"), ("mean", report_name)]
        )

    result = getattr(report.metrics, metric)(pos_label="A").reset_index()
    assert "Label / Average" not in result.columns
    result = result.set_index("Metric")
    for report_name in report.reports_:
        assert (
            result.loc[metric.capitalize(), ("mean", report_name)]
            == result_both_labels.loc[(metric.capitalize(), "A"), ("mean", report_name)]
        )

@pytest.mark.parametrize("response_method", ["predict", "predict_proba"])
def test_summarize_response_method_guidance(
    comparison_cross_validation_reports_regression,
    response_method,
):
    """`response_method` is not supported in `summarize` and should guide users."""
    report = comparison_cross_validation_reports_regression

    def business_loss(y_true_list, y_pred_list):
        loss = 0
        for y_true_, y_pred_ in zip(y_true_list, y_pred_list):
            # If under the market: 100% loss for me
            if y_true_ > y_pred_:
                loss = loss + float(y_true_ - y_pred_)
            # If I'm above the market, I waste time to sell
            # Each month costs 2k, and every month I lower the price by 10k
            else:
                loss = loss + float(2 * (y_pred_ - y_true_) / 10)
        return loss

    with pytest.raises(TypeError, match=r"custom_metric|make_scorer"):
        report.metrics.summarize(
            metric=business_loss,
            response_method=response_method,
        )

