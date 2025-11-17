import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils._testing import _convert_container

from skore import EstimatorReport
from skore._sklearn._plot.feature_importance.coefficients import CoefficientsDisplay


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_binary_classification_single_estimator(
    pyplot, logistic_binary_classification_with_train_test, fit_intercept
):
    """Check the attributes and default plotting behaviour of the coefficients plot with
    binary data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    estimator = clone(estimator).set_params(fit_intercept=fit_intercept)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.feature_importance.coefficients()
    assert isinstance(display, CoefficientsDisplay)

    expected_columns = [
        "estimator",
        "split",
        "feature",
        "label",
        "output",
        "coefficients",
    ]
    df = display.coefficients
    assert df.columns.tolist() == expected_columns
    for col in ("split", "output", "label"):
        assert df[col].isna().all()
    assert df["estimator"].nunique() == 1
    coef = np.concatenate([report.estimator_.intercept_, report.estimator_.coef_[0, :]])
    np.testing.assert_allclose(df["coefficients"], coef)

    df = display.frame()
    expected_columns = ["feature", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert df["feature"].tolist() == ["Intercept"] + columns_names
    if not fit_intercept:
        mask = df["feature"] == "Intercept"
        assert df.loc[mask, "coefficients"].item() == pytest.approx(0)

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_legend() is None
    assert display.ax_.get_title() == report.estimator_name_
    assert display.ax_.get_xlabel() == "Magnitude of coefficient"
    assert display.ax_.get_ylabel() == ""
