import matplotlib as mpl
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from skore import EstimatorReport


@pytest.mark.parametrize(
    "fixture_name, subplot_by, err_msg",
    [
        (
            "estimator_reports_binary_classification",
            "label",
            "No columns to group by.",
        ),
        (
            "estimator_reports_regression",
            "output",
            "No columns to group by.",
        ),
        (
            "estimator_reports_multiclass_classification",
            "incorrect",
            "Column incorrect not found in the frame. "
            + "It should be one of label, auto, None.",
        ),
        (
            "estimator_reports_multioutput_regression",
            "incorrect",
            "Column incorrect not found in the frame. "
            + "It should be one of output, auto, None.",
        ),
    ],
)
def test_invalid_subplot_by(pyplot, fixture_name, subplot_by, err_msg, request):
    reports = request.getfixturevalue(fixture_name)
    report = reports[0]
    display = report.inspection.coefficients()
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by=subplot_by)


@pytest.mark.parametrize(
    "fixture_name, subplot_by_tuples",
    [
        (
            "estimator_reports_binary_classification",
            [(None, 0)],
        ),
        (
            "estimator_reports_multiclass_classification",
            [("label", 3), (None, 0)],
        ),
        (
            "estimator_reports_regression",
            [(None, 0)],
        ),
        (
            "estimator_reports_multioutput_regression",
            [("output", 2), (None, 0)],
        ),
    ],
)
def test_valid_subplot_by(pyplot, fixture_name, subplot_by_tuples, request):
    """Check that we can pass non default values to `subplot_by`."""
    reports = request.getfixturevalue(fixture_name)
    report = reports[0]
    display = report.inspection.coefficients()
    for subplot_by, expected_len in subplot_by_tuples:
        display.plot(subplot_by=subplot_by)
        if subplot_by is None:
            assert isinstance(display.ax_, mpl.axes.Axes)
        else:
            assert len(display.ax_) == expected_len


def test_include_intercept_multioutput_fit_intercept_false(request):
    """fit_intercept=False multi-output: scalar intercept is repeated per output."""
    X_train, X_test, y_train, y_test = request.getfixturevalue(
        "multioutput_regression_train_test_split"
    )
    report = EstimatorReport(
        LinearRegression(fit_intercept=False),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.inspection.coefficients()
    frame = display.frame(include_intercept=True, format="long")
    intercept_rows = frame.query("feature == 'Intercept'")
    assert len(intercept_rows) == 2
    assert set(intercept_rows["output"].astype(str)) == {"0", "1"}
    np.testing.assert_array_equal(intercept_rows["coefficient"].values, [0.0, 0.0])
    assert display.frame(include_intercept=False).query("feature == 'Intercept'").empty
