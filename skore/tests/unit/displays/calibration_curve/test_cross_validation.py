import pandas as pd
import pytest

from skore import CalibrationDisplay


def test_unequal_fold_bins_remain_separate():
    calibration_report = pd.DataFrame(
        {
            "estimator": ["model"] * 5,
            "data_source": ["test"] * 5,
            "split": [0, 0, 0, 1, 1],
            "label": [1] * 5,
            "predicted_probability": [0.1, 0.4, 0.9, 0.2, 0.8],
            "fraction_of_positives": [0.0, 0.5, 1.0, 0.25, 0.75],
        }
    )
    display = CalibrationDisplay(
        calibration_report=calibration_report, report_type="cross-validation"
    )

    frame = display.frame(label=1)
    pd.testing.assert_frame_equal(
        frame.reset_index(drop=True),
        calibration_report.drop(columns=["estimator", "label"]),
    )

    ax = display.plot(label=1).axes[0]
    curves = [line for line in ax.lines if len(line.get_xdata()) > 0]
    assert sorted(len(line.get_xdata()) for line in curves) == [2, 2, 3]


@pytest.mark.parametrize(
    "fixture_name",
    [
        "cross_validation_reports_binary_classification",
        "cross_validation_reports_multiclass_classification",
    ],
)
def test_plot_folds_by_label(fixture_name, request):
    report = request.getfixturevalue(fixture_name)[0]
    display = report.inspection.calibration_curve(n_bins=5, strategy="uniform")

    fig = display.plot(label=None)
    ax = fig.axes[0]

    assert ax.get_xlabel() == "Mean predicted probability"
    assert ax.get_ylabel() == "Fraction of positives"

    curves = [
        line for line in ax.lines if line.get_alpha() == 0.4 and len(line.get_xdata())
    ]
    assert len(curves) == len(display.labels) * len(report.reports_)
    colors = [line.get_color() for line in curves]
    for label_index in range(len(display.labels)):
        label_colors = colors[
            label_index * len(report.reports_) : (label_index + 1)
            * len(report.reports_)
        ]
        assert len(set(label_colors)) == 1
    assert len(set(colors)) == len(display.labels)
    assert len({line.get_linestyle() for line in curves}) == 1
