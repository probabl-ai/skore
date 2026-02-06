import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from skore import CoefficientsDisplay, CrossValidationReport, EstimatorReport


@pytest.mark.parametrize("method", ["frame", "plot"])
def test_coefficients_display_invalid_report_type(pyplot, method):
    """Check that CoefficientsDisplay raises TypeError for invalid `report_type`."""
    coefficients = pd.DataFrame(
        {
            "estimator": ["estimator1"],
            "split": [0],
            "feature": ["feature1"],
            "label": [np.nan],
            "output": [np.nan],
            "coefficients": [1.0],
        }
    )

    display = CoefficientsDisplay(coefficients=coefficients, report_type="invalid-type")
    with pytest.raises(TypeError, match="Unexpected report type: 'invalid-type'"):
        getattr(display, method)()


def test_coefficients_display_barplot_kwargs(
    pyplot, logistic_binary_classification_with_train_test
):
    """Check that custom `barplot_kwargs` are applied to `EstimatorReport` plots."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.coefficients()
    result = display.set_style(barplot_kwargs={"color": "red"}).plot()

    patches = result.axes[0, 0].patches
    assert len(patches) > 0
    for patch in patches:
        assert patch.get_facecolor() == (0.875, 0.125, 0.125, 1.0)  # red in RGBA


def test_coefficients_display_boxplot_kwargs(pyplot):
    """Check that custom boxplot_kwargs are applied to CrossValidationReport plots."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    splitter = 3

    report = CrossValidationReport(
        LogisticRegression(random_state=42), X, y, splitter=splitter
    )

    display = report.inspection.coefficients()
    result = display.set_style(
        boxplot_kwargs={"boxprops": {"facecolor": "blue"}},
        stripplot_kwargs={"color": "red"},
    ).plot()

    patches = result.axes[0, 0].patches
    assert len(patches) > 0
    for patch in patches:
        assert patch.get_facecolor() == (0.0, 0.0, 1.0, 1.0)  # blue in RGBA
    expected_red = np.array([1, 0, 0, 0.5])  # red in RGBA
    for collection in result.axes[0, 0].collections:
        for facecolor in collection.get_facecolor():
            np.testing.assert_allclose(facecolor, expected_red)
