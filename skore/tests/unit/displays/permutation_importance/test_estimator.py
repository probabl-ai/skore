import matplotlib as mpl
import numpy as np
import pytest
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from skore._utils._testing import custom_r2_score


@pytest.mark.parametrize(
    "fixture_name, subplot_by, err_msg",
    [
        (
            "estimator_reports_binary_classification",
            "label",
            "Cannot use subplot_by='label'.*does not provide per-label",
        ),
        (
            "estimator_reports_regression",
            "output",
            "Cannot use subplot_by='output'.*does not provide per-output",
        ),
        (
            "estimator_reports_regression",
            "unknown",
            "The column 'unknown' is not available for subplotting",
        ),
    ],
)
def test_invalid_subplot_by(pyplot, fixture_name, subplot_by, err_msg, request):
    report = request.getfixturevalue(fixture_name)[0]
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by=subplot_by)


@pytest.mark.parametrize(
    "fixture_name, metric, metric_name, subplot_by_tuples",
    [
        (
            "estimator_reports_binary_classification",
            None,
            None,
            [(None, 0), ("auto", 0)],
        ),
        (
            "estimator_reports_multiclass_classification",
            make_scorer(precision_score, average=None),
            "precision score",
            [("label", 3), ("auto", 3), (None, 0)],
        ),
        (
            "estimator_reports_regression",
            None,
            None,
            [(None, 0), ("auto", 0)],
        ),
        (
            "estimator_reports_multioutput_regression",
            make_scorer(r2_score, multioutput="raw_values"),
            "r2 score",
            [("output", 2), ("auto", 2), (None, 0)],
        ),
    ],
)
def test_valid_subplot_by(
    pyplot, fixture_name, metric, metric_name, subplot_by_tuples, request
):
    report = request.getfixturevalue(fixture_name)[0]
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metric
    )
    for subplot_by, expected_len in subplot_by_tuples:
        display.plot(metric=metric_name, subplot_by=subplot_by)
        if expected_len == 0:
            assert isinstance(display.ax_, mpl.axes.Axes)
        else:
            assert len(display.ax_.flatten()) == expected_len


def test_multiple_metrics_require_metric_param(pyplot, estimator_reports_regression):
    report = estimator_reports_regression[0]
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=["r2", "neg_mean_squared_error"]
    )
    with pytest.raises(ValueError, match="Please select a metric"):
        display.plot()

    display.plot(metric="r2")
    assert display.ax_.get_xlabel() == "Decrease in r2"
    display.plot(metric="neg_mean_squared_error")
    assert display.ax_.get_xlabel() == "Decrease in neg_mean_squared_error"


def test_frame_metric_filter(estimator_reports_regression):
    report = estimator_reports_regression[0]
    display = report.inspection.permutation_importance(
        n_repeats=2,
        seed=0,
        metric=["r2", "neg_mean_squared_error"],
    )
    assert set(display.frame()["metric"].unique()) == {"r2", "neg_mean_squared_error"}
    assert set(display.frame(metric="r2")["metric"].unique()) == {"r2"}
    assert set(display.frame(metric=["r2"])["metric"].unique()) == {"r2"}


def test_callable_metric_name(pyplot, estimator_reports_regression):
    report = estimator_reports_regression[0]
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=custom_r2_score
    )
    display.plot(metric="custom r2 score")
    assert display.ax_.get_xlabel() == "Decrease in custom r2 score"


def test_per_label_metrics_internal_data(
    estimator_reports_multiclass_classification,
):
    report = estimator_reports_multiclass_classification[0]
    metrics = {
        "precision": make_scorer(precision_score, average=None),
        "recall": make_scorer(recall_score, average=None),
    }
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metrics
    )
    df = display.importances
    np.testing.assert_array_equal(df["label"].unique(), report.estimator_.classes_)
    assert df["output"].isna().all()
    assert set(df["metric"].unique()) == {"precision", "recall"}

    frame = display.frame()
    assert "label" in frame.columns


def test_per_output_metrics_internal_data(
    estimator_reports_multioutput_regression,
):
    report = estimator_reports_multioutput_regression[0]
    metric = {
        "r2": make_scorer(r2_score, multioutput="raw_values"),
        "mse": make_scorer(mean_squared_error, multioutput="raw_values"),
    }
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metric
    )
    df = display.importances
    assert df["label"].isna().all()
    assert set(df["output"].unique()) == {0, 1}
    assert set(df["metric"].unique()) == {"r2", "mse"}

    frame = display.frame()
    assert "output" in frame.columns


@pytest.mark.parametrize("aggregate", [None, ("mean", "std")])
def test_frame_mixed_averaged_and_non_averaged_metrics(
    estimator_reports_binary_classification, aggregate
):
    report = estimator_reports_binary_classification[0]
    metrics = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average=None),
    }
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metrics
    )
    frame = display.frame(aggregate=aggregate)

    assert set(frame["metric"].unique()) == {"accuracy", "precision"}
    assert "label" in frame.columns
    assert frame.query("metric == 'accuracy'")["label"].isna().all()
    assert not frame.query("metric == 'precision'")["label"].isna().any()


def test_plot_mixed_averaged_and_non_averaged_metrics(
    pyplot, estimator_reports_binary_classification
):
    report = estimator_reports_binary_classification[0]
    metrics = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average=None),
    }
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metrics
    )
    display.plot(metric="accuracy")
    assert hasattr(display, "figure_")
    display.plot(metric="precision")
    assert hasattr(display, "figure_")


def test_default_metric_name_classifier(estimator_reports_binary_classification):
    report = estimator_reports_binary_classification[0]
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    assert display.importances["metric"].unique() == ["accuracy"]


def test_default_metric_name_regressor(estimator_reports_regression):
    report = estimator_reports_regression[0]
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    assert display.importances["metric"].unique() == ["r2"]


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_data_source(estimator_reports_binary_classification, data_source):
    report = estimator_reports_binary_classification[0]
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, data_source=data_source
    )
    assert display.importances["data_source"].unique() == [data_source]
