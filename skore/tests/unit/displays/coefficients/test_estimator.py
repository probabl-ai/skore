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

    expected_columns = ["estimator_name", "split", "feature_name", "coefficients"]
    assert display.coefficients.columns.tolist() == expected_columns

    df = display.frame()
    expected_columns = ["feature_name", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert df["feature_name"].tolist() == ["Intercept"] + columns_names
