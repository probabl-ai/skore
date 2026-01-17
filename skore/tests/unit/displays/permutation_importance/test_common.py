import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from skore import PermutationImportanceDisplay, EstimatorReport


@pytest.mark.parametrize("method", ["frame", "plot"])
def test_permutation_importance_display_invalid_report_type(pyplot, method):
    """Check that PermutationImportanceDisplay raises TypeError for invalid
    `report_type`."""
    importances = pd.DataFrame(
        {
            "data_source": ["test"],
            "metric": ["r2"],
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
        getattr(display, method)()


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

    display = report.feature_importance.permutation()
    display.set_style(
        boxplot_kwargs={"boxprops": {"facecolor": "blue"}},
        stripplot_kwargs={"color": "red"},
    ).plot()

    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    patches = display.ax_.patches
    for patch in patches:
        assert patch.get_facecolor() == (0.0, 0.0, 1.0, 1.0)  # blue in RGBA
    expected_red = np.array([1, 0, 0, 0.5])  # red in RGBA
    for collection in display.ax_.collections:
        for facecolor in collection.get_facecolor():
            np.testing.assert_allclose(facecolor, expected_red)
