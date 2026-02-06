import matplotlib as mpl
import pytest


@pytest.mark.parametrize(
    "fixture_name, valid_values",
    [
        (
            "comparison_estimator_reports_binary_classification",
            ["estimator", "auto", "None"],
        ),
        (
            "comparison_estimator_reports_multiclass_classification",
            ["estimator", "label", "auto"],
        ),
        (
            "comparison_estimator_reports_regression",
            ["estimator", "auto", "None"],
        ),
        (
            "comparison_estimator_reports_multioutput_regression",
            ["estimator", "output", "auto"],
        ),
    ],
)
def test_invalid_subplot_by(pyplot, fixture_name, valid_values, request):
    report = request.getfixturevalue(fixture_name)
    display = report.inspection.coefficients()
    err_msg = (
        "Column incorrect not found in the frame."
        f" It should be one of {', '.join(valid_values)}."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="incorrect")


@pytest.mark.parametrize(
    "fixture_name, subplot_by_tuples",
    [
        (
            "comparison_estimator_reports_binary_classification",
            [(None, 0), ("estimator", 2)],
        ),
        (
            "comparison_estimator_reports_multiclass_classification",
            [("label", 3), ("estimator", 2)],
        ),
        (
            "comparison_estimator_reports_regression",
            [(None, 0), ("estimator", 2)],
        ),
        (
            "comparison_estimator_reports_multioutput_regression",
            [("output", 2), ("estimator", 2)],
        ),
    ],
)
def test_valid_subplot_by(pyplot, fixture_name, subplot_by_tuples, request):
    report = request.getfixturevalue(fixture_name)
    display = report.inspection.coefficients()
    for subplot_by, expected_len in subplot_by_tuples:
        display.plot(subplot_by=subplot_by)
        if subplot_by is None:
            assert isinstance(display.ax_, mpl.axes.Axes)
        else:
            assert len(display.ax_) == expected_len


@pytest.mark.parametrize(
    "fixture_name",
    [
        "comparison_estimator_reports_multiclass_classification",
        "comparison_estimator_reports_multioutput_regression",
    ],
)
def test_subplot_by_none_multiclass_or_multioutput(
    pyplot,
    request,
    fixture_name,
):
    """Check that an error is raised when `subplot_by=None` and there are multiple
    labels (multiclass) or outputs (multi-output regression)."""
    report = request.getfixturevalue(fixture_name)
    display = report.inspection.coefficients()

    err_msg = (
        "There are multiple labels or outputs and `subplot_by` is `None`. "
        "There is too much information to display on a single plot. "
        "Please provide a column to group by using `subplot_by`."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by=None)


@pytest.mark.parametrize(
    "fixture_name, subplot_by",
    [
        (
            "comparison_estimator_reports_binary_classification_different_features",
            None,
        ),
        (
            "comparison_estimator_reports_multiclass_classification_different_features",
            "label",
        ),
    ],
)
def test_different_features(pyplot, fixture_name, subplot_by, request):
    """Check that we get a proper report even if the estimators do not have the same
    input features."""
    report = request.getfixturevalue(fixture_name)
    display = report.inspection.coefficients()

    err_msg = (
        "The estimators have different features and should be plotted on different "
        "axis using `subplot_by='estimator'`."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by=subplot_by)

    display.plot(subplot_by="estimator")
    assert hasattr(display, "facet_")
