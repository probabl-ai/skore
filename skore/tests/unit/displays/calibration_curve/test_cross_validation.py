import pytest


@pytest.mark.parametrize(
    "fixture_name, aggregate, err_msg",
    [
        (
            "cross_validation_reports_binary_classification",
            "mean",
            "aggregate must be None or",
        ),
        (
            "cross_validation_reports_binary_classification",
            ("std", "mean"),
            "aggregate must be None or",
        ),
    ],
)
def test_invalid_aggregate(fixture_name, aggregate, err_msg, request):
    """Check that invalid aggregate values raise a ValueError."""
    report = request.getfixturevalue(fixture_name)[0]
    display = report.inspection.calibration_curve(n_bins=5, strategy="uniform")
    with pytest.raises(ValueError, match=err_msg):
        display.frame(aggregate=aggregate)


@pytest.mark.parametrize(
    "fixture_name, aggregate",
    [
        (
            "cross_validation_reports_binary_classification",
            None,
        ),
        (
            "cross_validation_reports_binary_classification",
            ("mean", "std"),
        ),
        (
            "cross_validation_reports_multiclass_classification",
            None,
        ),
        (
            "cross_validation_reports_multiclass_classification",
            ("mean", "std"),
        ),
    ],
)
def test_plot_aggregate(pyplot, fixture_name, aggregate, request):
    """Check that plot works for both aggregate modes across classification tasks."""
    report = request.getfixturevalue(fixture_name)[0]
    display = report.inspection.calibration_curve(n_bins=5, strategy="uniform")

    fig = display.plot(aggregate=aggregate)
    ax = fig.axes[0]

    assert ax.get_xlabel() == "Mean predicted probability"
    assert ax.get_ylabel() == "Fraction of positives"
