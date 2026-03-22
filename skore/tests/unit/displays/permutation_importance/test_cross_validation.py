import matplotlib as mpl
import pytest
from sklearn.metrics import make_scorer, precision_score, r2_score


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
    report = request.getfixturevalue(f"cross_validation_reports_{task}")[0]
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
        ("split", 2),
        (None, 1),
        ("auto", 1),
    ],
)
def test_valid_subplot_by(pyplot, task, subplot_by, expected_len, request):
    report = request.getfixturevalue(f"cross_validation_reports_{task}")[0]
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    fig = display.plot(subplot_by=subplot_by)
    axes = fig.axes
    if expected_len == 1:
        assert isinstance(axes[0], mpl.axes.Axes)
    else:
        assert len(axes) == expected_len


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
    report = request.getfixturevalue(f"cross_validation_reports_{task}")[0]
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metric
    )
    fig = display.plot(metric=metric_name, subplot_by=subplot_by)
    axes = fig.axes
    assert len(axes) == expected_len

    valid_values = [subplot_by, "split", "auto", "None"]
    err_msg = (
        f"The column 'invalid' is not available for subplotting. You can use the "
        f"following values to create subplots: {', '.join(valid_values)}"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize(
    "level, expected_columns",
    [
        ("splits", ["value_mean", "value_std"]),
        ("repetitions", ["split", "value_mean", "value_std"]),
    ],
)
def test_frame_aggregation_level(
    cross_validation_reports_binary_classification,
    level,
    expected_columns,
):
    report = cross_validation_reports_binary_classification[0]
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    frame = display.frame(level=level)
    assert set(frame.columns) >= set(expected_columns)


def test_frame_metric_filter(cross_validation_reports_regression):
    report = cross_validation_reports_regression[0]
    display = report.inspection.permutation_importance(
        n_repeats=2,
        seed=0,
        metric=["r2", "neg_mean_squared_error"],
    )
    assert set(display.frame()["metric"]) == {"r2", "neg_mean_squared_error"}
    assert set(display.frame(metric="r2")["metric"]) == {"r2"}


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_data_source(cross_validation_reports_binary_classification, data_source):
    report = cross_validation_reports_binary_classification[0]
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, data_source=data_source
    )
    assert set(display.importances["data_source"]) == {data_source}
