import matplotlib as mpl
import pytest
from sklearn.metrics import make_scorer, precision_score


@pytest.mark.parametrize(
    "fixture_name, subplot_by, err_msg",
    [
        (
            "cross_validation_reports_binary_classification",
            "label",
            "Cannot use subplot_by='label'.*does not provide per-label",
        ),
        (
            "cross_validation_reports_binary_classification",
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
            "cross_validation_reports_binary_classification",
            None,
            None,
            [("split", 2), ("auto", 0), (None, 0)],
        ),
        (
            "cross_validation_reports_multiclass_classification",
            make_scorer(precision_score, average=None),
            "precision score",
            [("label", 3), ("auto", 3)],
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
        title = display.figure_.get_suptitle()
        if subplot_by == "split":
            assert "averaged over splits" not in title
        else:
            assert "averaged over splits" in title


def test_frame_has_split_column(cross_validation_reports_binary_classification):
    report = cross_validation_reports_binary_classification[0]
    display = report.inspection.permutation_importance(n_repeats=2, seed=0)
    frame = display.frame()
    assert "split" in frame.columns
    assert frame["split"].nunique() == len(report.estimator_reports_)


def test_frame_metric_filter(cross_validation_reports_regression):
    report = cross_validation_reports_regression[0]
    display = report.inspection.permutation_importance(
        n_repeats=2,
        seed=0,
        metric=["r2", "neg_mean_squared_error"],
    )
    assert set(display.frame()["metric"].unique()) == {"r2", "neg_mean_squared_error"}
    assert set(display.frame(metric="r2")["metric"].unique()) == {"r2"}


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_data_source(cross_validation_reports_binary_classification, data_source):
    report = cross_validation_reports_binary_classification[0]
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, data_source=data_source
    )
    assert display.importances["data_source"].unique() == [data_source]
