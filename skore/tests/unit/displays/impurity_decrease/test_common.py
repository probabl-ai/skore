import pandas as pd
import pytest

from skore import EstimatorReport, ImpurityDecreaseDisplay


@pytest.mark.parametrize("method", ["frame", "plot"])
def test_impurity_decrease_display_invalid_report_type(pyplot, method):
    """Check that ImpurityDecreaseDisplay raises TypeError for invalid `report_type`."""
    importances = pd.DataFrame(
        {
            "estimator": ["estimator1"],
            "feature": ["feature1"],
            "importances": [1.0],
        }
    )

    display = ImpurityDecreaseDisplay(
        importances=importances, report_type="invalid-type"
    )
    with pytest.raises(TypeError, match="Unexpected report type: 'invalid-type'"):
        getattr(display, method)()


def test_impurity_decrease_display_barplot_kwargs(
    pyplot, forest_binary_classification_with_train_test
):
    """Check that custom `barplot_kwargs` are applied to `EstimatorReport` plots."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.impurity_decrease()
    display.set_style(barplot_kwargs={"color": "red"}).plot()

    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    patches = display.ax_.patches
    assert len(patches) > 0
    for patch in patches:
        assert patch.get_facecolor() == (0.875, 0.125, 0.125, 1.0)  # red in RGBA
