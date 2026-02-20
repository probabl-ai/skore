import matplotlib as mpl
import pytest
from sklearn.metrics import make_scorer, precision_score, r2_score


@pytest.mark.parametrize(
    "fixture_name, valid_values",
    [
        (
            "comparison_cross_validation_reports_binary_classification",
            ["estimator", "split", "auto", "None"],
        ),
        (
            "comparison_cross_validation_reports_multiclass_classification",
            ["estimator", "split", "auto", "None"],
        ),
        (
            "comparison_cross_validation_reports_regression",
            ["estimator", "split", "auto", "None"],
        ),
        (
            "comparison_cross_validation_reports_multioutput_regression",
            ["estimator", "split", "auto", "None"],
        ),
    ],
)
def test_invalid_subplot_by(pyplot, fixture_name, valid_values, request):
    report = request.getfixturevalue(fixture_name)
    display = report.inspection.permutation_importance(seed=0, n_repeats=2)
    err_msg = (
        "The column 'incorrect' is not available for subplotting. You can use "
        f"the following values to create subplots: {', '.join(valid_values)}"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="incorrect")


def test_valid_subplot_by(pyplot, comparison_cross_validation_reports_regression):
    report = comparison_cross_validation_reports_regression
    display = report.inspection.permutation_importance(seed=0, n_repeats=2)
    display.plot(subplot_by=None)
    assert isinstance(display.ax_, mpl.axes.Axes)

    display.plot(subplot_by="split")
    assert len(display.ax_) == 2

    display.plot(subplot_by="estimator")
    assert len(display.ax_) == 2


@pytest.mark.parametrize(
    "fixture_name, metric, metric_name, subplot_by",
    [
        (
            "comparison_cross_validation_reports_multiclass_classification",
            make_scorer(precision_score, average=None),
            "precision score",
            "label",
        ),
        (
            "comparison_cross_validation_reports_multioutput_regression",
            make_scorer(r2_score, multioutput="raw_values"),
            "r2 score",
            "output",
        ),
    ],
)
def test_subplot_by_none_multiclass_or_multioutput(
    pyplot, fixture_name, metric, metric_name, subplot_by, request
):
    report = request.getfixturevalue(fixture_name)
    display = report.inspection.permutation_importance(
        seed=0, n_repeats=2, metric=metric
    )
    err_msg = (
        "There are multiple labels or outputs and `subplot_by` is `None`. "
        "There is too much information to display on a single plot. "
        "Please provide a column to group by using `subplot_by`."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(metric=metric_name, subplot_by=None)

    display.plot(metric=metric_name, subplot_by=subplot_by)
    assert len(display.ax_) > 1


@pytest.mark.parametrize(
    "fixture_name, subplot_by",
    [
        (
            "comparison_cross_validation_reports_binary_classification_different_features",
            None,
        ),
        (
            "comparison_cross_validation_reports_multiclass_classification_different_features",
            "split",
        ),
    ],
)
def test_different_features(pyplot, fixture_name, subplot_by, request):
    report = request.getfixturevalue(fixture_name)
    display = report.inspection.permutation_importance(seed=0, n_repeats=2)
    err_msg = (
        "The estimators have different features and should be plotted on different "
        "axis using `subplot_by='estimator'`."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by=subplot_by)

    display.plot(subplot_by="estimator")
    assert hasattr(display, "facet_")
