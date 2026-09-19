import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
import skrub
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skore import CrossValidationReport


@pytest.mark.parametrize(
    "fixture_name, subplot_by, err_msg",
    [
        (
            "cross_validation_reports_binary_classification",
            "label",
            "No columns to group by.",
        ),
        (
            "cross_validation_reports_regression",
            "output",
            "No columns to group by.",
        ),
        (
            "cross_validation_reports_multiclass_classification",
            "incorrect",
            "Column incorrect not found in the frame."
            + " It should be one of label, auto, None.",
        ),
        (
            "cross_validation_reports_multioutput_regression",
            "incorrect",
            "Column incorrect not found in the frame."
            + " It should be one of output, auto, None.",
        ),
    ],
)
def test_invalid_subplot_by(fixture_name, subplot_by, err_msg, request):
    reports = request.getfixturevalue(fixture_name)
    report = reports[0]
    display = report.inspection.coefficients()
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by=subplot_by)


@pytest.mark.parametrize(
    "fixture_name, subplot_by_tuples",
    [
        (
            "cross_validation_reports_binary_classification",
            [(None, 0)],
        ),
        (
            "cross_validation_reports_multiclass_classification",
            [("label", 3), (None, 0)],
        ),
        (
            "cross_validation_reports_regression",
            [(None, 0)],
        ),
        (
            "cross_validation_reports_multioutput_regression",
            [("output", 2), (None, 0)],
        ),
    ],
)
def test_valid_subplot_by(fixture_name, subplot_by_tuples, request):
    """Check that we can pass non default values to `subplot_by`."""
    reports = request.getfixturevalue(fixture_name)
    report = reports[0]
    display = report.inspection.coefficients()
    for subplot_by, expected_len in subplot_by_tuples:
        fig = display.plot(subplot_by=subplot_by)
        axes = fig.axes
        if subplot_by is None:
            assert len(axes) == 1
            assert isinstance(axes[0], mpl.axes.Axes)
        else:
            assert len(axes) == expected_len


def test_scale_features(cross_validation_reports_regression):
    """CV displays store per-split feature_std and can scale coefficients."""
    report = cross_validation_reports_regression[0]
    display = report.inspection.coefficients()
    assert display.coefficients["feature_std"].notna().all()

    raw = display.frame(aggregate=None, include_intercept=False)
    scaled = display.frame(aggregate=None, include_intercept=False, scale_features=True)
    assert "feature_std" not in scaled.columns

    feature_std = display.coefficients.query("feature != 'Intercept'")["feature_std"]
    np.testing.assert_allclose(
        scaled["coefficient"], raw["coefficient"] * feature_std.to_numpy()
    )


def test_skrub_learner_matches_sklearn_pipeline():
    """A skrub learner gives the same per-split coefficients as the equivalent
    scikit-learn pipeline."""
    X, y = make_regression(n_samples=60, n_features=3, random_state=0)
    X = pd.DataFrame(X, columns=["a", "b", "c"])
    data_op = skrub.X().skb.apply(StandardScaler()).skb.apply(Ridge(), y=skrub.y())
    skrub_report = CrossValidationReport(
        data_op.skb.make_learner(), data={"X": X, "y": y}, splitter=3
    )
    sklearn_report = CrossValidationReport(
        make_pipeline(StandardScaler(), Ridge()), X, y, splitter=3
    )
    pd.testing.assert_frame_equal(
        skrub_report.inspection.coefficients().frame(
            aggregate=None, scale_features=True
        ),
        sklearn_report.inspection.coefficients().frame(
            aggregate=None, scale_features=True
        ),
    )
