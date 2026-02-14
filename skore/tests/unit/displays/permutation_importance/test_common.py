import numpy as np
import pandas as pd
import pytest

from skore import EstimatorReport, PermutationImportanceDisplay


@pytest.mark.parametrize("method", ["frame", "plot"])
def test_permutation_importance_display_invalid_report_type(pyplot, method):
    """Check that PermutationImportanceDisplay raises TypeError for invalid
    `report_type`."""
    importances = pd.DataFrame(
        {
            "estimator": ["estimator"],
            "data_source": ["test"],
            "metric": ["r2"],
            "split": [np.nan],
            "feature": ["feature1"],
            "label": [np.nan],
            "output": [np.nan],
            "repetition": [0],
            "value": [1.0],
        }
    )

    display = PermutationImportanceDisplay(
        importances=importances, report_type="invalid-type"
    )
    with pytest.raises(TypeError, match="Unexpected report type: 'invalid-type'"):
        getattr(display, method)(metric="r2")


def test_permutation_importance_display_barplot_kwargs(
    pyplot, logistic_binary_classification_with_train_test
):
    """Check that custom `barplot_kwargs` are applied to `EstimatorReport` plots."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.permutation_importance()
    display.set_style(
        boxplot_kwargs={"boxprops": {"facecolor": "blue"}},
        stripplot_kwargs={"color": "red"},
    ).plot(metric="accuracy")

    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    patches = display.ax_.patches
    for patch in patches:
        assert patch.get_facecolor() == (0.0, 0.0, 1.0, 1.0)  # blue in RGBA
    expected_red = np.array([1, 0, 0, 0.5])  # red in RGBA
    for collection in display.ax_.collections:
        for facecolor in collection.get_facecolor():
            np.testing.assert_allclose(facecolor, expected_red)


def test_set_style_with_single_kwarg(
    pyplot, logistic_binary_classification_with_train_test
):
    """Check that set_style works when only one kwarg is passed."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.permutation_importance()
    display.set_style(stripplot_kwargs={"alpha": 0.8}).plot(metric="accuracy")
    for collection in display.ax_.collections:
        assert collection.get_alpha() == 0.8
