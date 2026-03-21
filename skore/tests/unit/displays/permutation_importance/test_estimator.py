import matplotlib as mpl
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
    "task",
    [
        "binary_classification",
        "multiclass_classification",
        "regression",
        "multioutput_regression",
    ],
)
def test_invalid_subplot_by(pyplot, task, request):
    report = request.getfixturevalue(f"estimator_reports_{task}")[0]
    err_msg = "The column 'invalid' is not available for subplotting."
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize(
    "task",
    [
        "binary_classification",
        "multiclass_classification",
        "regression",
        "multioutput_regression",
    ],
)
@pytest.mark.parametrize(
    "subplot_by, expected_len",
    [
        (None, 1),
        ("auto", 1),
    ],
)
def test_valid_subplot_by(pyplot, task, subplot_by, expected_len, request):
    report = request.getfixturevalue(f"estimator_reports_{task}")[0]
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    display.plot(subplot_by=subplot_by)
    if expected_len == 1:
        assert isinstance(display.ax_, mpl.axes.Axes)
    else:
        assert len(display.ax_.flatten()) == expected_len


@pytest.mark.parametrize(
    "task, metric, metric_name, subplot_by, expected_len",
    [
        (
            "multiclass_classification",
            make_scorer(precision_score, average=None),
            "precision score",
            "label",
            3,
        ),
        (
            "multioutput_regression",
            make_scorer(r2_score, multioutput="raw_values"),
            "r2 score",
            "output",
            2,
        ),
    ],
)
def test_subplot_by_non_averaged_metrics(
    pyplot, task, metric, metric_name, subplot_by, expected_len, request
):
    report = request.getfixturevalue(f"estimator_reports_{task}")[0]
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metric
    )
    display.plot(metric=metric_name, subplot_by=subplot_by)
    assert len(display.ax_) == expected_len

    valid_values = [subplot_by, "auto", "None"]
    err_msg = (
        f"The column 'invalid' is not available for subplotting. You can use the "
        f"following values to create subplots: {', '.join(valid_values)}"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


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
    assert set(display.frame()["metric"]) == {"r2", "neg_mean_squared_error"}
    assert set(display.frame(metric="r2")["metric"]) == {"r2"}
    assert set(display.frame(metric=["r2"])["metric"]) == {"r2"}


def test_callable_metric_name(pyplot, estimator_reports_regression):
    report = estimator_reports_regression[0]
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=custom_r2_score
    )
    display.plot(metric="custom r2 score")
    assert display.ax_.get_xlabel() == "Decrease in custom r2 score"


def test_per_label_metrics_frame(
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
    frame = display.frame()
    assert set(frame["label"]) == {0, 1, 2}
    assert set(frame["metric"]) == {"precision", "recall"}

    assert "output" not in frame.columns
    assert "label" in frame.columns


def test_per_output_metrics_frame(
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
    frame = display.frame()
    assert set(frame["output"]) == {0, 1}
    assert set(frame["metric"]) == {"r2", "mse"}

    assert "output" in frame.columns
    assert "label" not in frame.columns


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

    assert set(frame["metric"]) == {"accuracy", "precision"}
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
    assert set(display.importances["metric"]) == {"accuracy"}


def test_default_metric_name_regressor(estimator_reports_regression):
    report = estimator_reports_regression[0]
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    assert set(display.importances["metric"]) == {"r2"}


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_data_source(estimator_reports_binary_classification, data_source):
    report = estimator_reports_binary_classification[0]
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, data_source=data_source
    )
    assert set(display.importances["data_source"]) == {data_source}
