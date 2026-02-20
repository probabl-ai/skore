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
    report = request.getfixturevalue(f"comparison_estimator_reports_{task}")
    display = report.inspection.permutation_importance(seed=0, n_repeats=2)
    err_msg = (
        "The column 'incorrect' is not available for subplotting. You can use "
        f"the following values to create subplots: {
            ', '.join(
                ['estimator', 'auto', 'None'],
            )
        }"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="incorrect")


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
        ("estimator", 2),
        ("auto", 0),
        (None, 0),
    ],
)
def test_valid_subplot_by(pyplot, task, subplot_by, expected_len, request):
    report = request.getfixturevalue(f"comparison_estimator_reports_{task}")
    display = report.inspection.permutation_importance(seed=0, n_repeats=2)
    display.plot(subplot_by=subplot_by)
    if expected_len == 0:
        assert isinstance(display.ax_, mpl.axes.Axes)
    else:
        assert len(display.ax_.flatten()) == expected_len


@pytest.mark.parametrize(
    "task",
    [
        "binary_classification",
        "multiclass_classification",
        "regression",
        "multioutput_regression",
    ],
)
def test_different_features(pyplot, task, request):
    report = request.getfixturevalue(
        f"comparison_estimator_reports_{task}_different_features"
    )
    display = report.inspection.permutation_importance(seed=0, n_repeats=2)
    err_msg = (
        "The estimators have different features and should be plotted on different "
        "axis using `subplot_by='estimator'`."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by=None)

    display.plot(subplot_by="estimator")
    assert len(display.ax_) == 2


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
    report = request.getfixturevalue(f"comparison_estimator_reports_{task}")
    display = report.inspection.permutation_importance(
        n_repeats=2, seed=0, metric=metric
    )
    display.plot(metric=metric_name, subplot_by=subplot_by)
    assert len(display.ax_) == expected_len

    valid_values = ["estimator", subplot_by, "auto", "None"]
    err_msg = (
        f"The column 'invalid' is not available for subplotting. You can use the "
        f"following values to create subplots: {', '.join(valid_values)}"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")
