import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import _convert_container

from skore import EstimatorReport, ImpurityDecreaseDisplay


@pytest.mark.parametrize("with_preprocessing", [True, False])
def test_binary_classification(
    pyplot,
    forest_binary_classification_with_train_test,
    with_preprocessing,
):
    """Check the attributes and default plotting behaviour of the impurity decrease plot
    with binary data."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    predictor = clone(estimator)
    if with_preprocessing:
        model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])
    else:
        model = predictor

    report = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.impurity_decrease()
    assert isinstance(display, ImpurityDecreaseDisplay)

    expected_columns = ["estimator", "split", "feature", "importance"]
    df = display.importances
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    assert df["estimator"].nunique() == 1

    fitted_predictor = report.estimator_
    if with_preprocessing:
        fitted_predictor = fitted_predictor.named_steps["predictor"]
    feature_importances = fitted_predictor.feature_importances_
    np.testing.assert_allclose(df["importance"].to_numpy(), feature_importances)

    df = display.frame()
    expected_columns = ["feature", "importance"]
    assert df.columns.tolist() == expected_columns
    assert df["feature"].tolist() == columns_names

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_xlabel() == "Mean Decrease in Impurity (MDI)"
    assert display.ax_.get_ylabel() == ""
    estimator_name = display.importances["estimator"][0]
    assert (
        display.figure_.get_suptitle()
        == f"Mean Decrease in Impurity (MDI) of {estimator_name}"
    )


@pytest.mark.parametrize("with_preprocessing", [True, False])
def test_multiclass_classification(
    pyplot,
    forest_multiclass_classification_with_train_test,
    with_preprocessing,
):
    """Check the attributes and default plotting behaviour of the impurity decrease plot
    with multiclass data."""
    estimator, X_train, X_test, y_train, y_test = (
        forest_multiclass_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    predictor = clone(estimator)
    if with_preprocessing:
        model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])
    else:
        model = predictor

    report = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.impurity_decrease()
    assert isinstance(display, ImpurityDecreaseDisplay)

    expected_columns = ["estimator", "split", "feature", "importance"]
    df = display.importances
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    assert df["estimator"].nunique() == 1

    fitted_predictor = report.estimator_
    if with_preprocessing:
        fitted_predictor = fitted_predictor.named_steps["predictor"]
    feature_importances = fitted_predictor.feature_importances_
    np.testing.assert_allclose(df["importance"].to_numpy(), feature_importances)

    df = display.frame()
    expected_columns = ["feature", "importance"]
    assert df.columns.tolist() == expected_columns
    assert df["feature"].tolist() == columns_names

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_xlabel() == "Mean Decrease in Impurity (MDI)"
    assert display.ax_.get_ylabel() == ""
    estimator_name = display.importances["estimator"][0]
    assert (
        display.figure_.get_suptitle()
        == f"Mean Decrease in Impurity (MDI) of {estimator_name}"
    )


@pytest.mark.parametrize("with_preprocessing", [True, False])
def test_single_output_regression(
    pyplot,
    regression_train_test_split,
    with_preprocessing,
):
    """Check the attributes and default plotting behaviour of the impurity decrease plot
    with single output regression data."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    predictor = RandomForestRegressor(random_state=42)
    if with_preprocessing:
        model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])
    else:
        model = predictor

    report = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.impurity_decrease()
    assert isinstance(display, ImpurityDecreaseDisplay)

    expected_columns = ["estimator", "split", "feature", "importance"]
    df = display.importances
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    assert df["estimator"].nunique() == 1

    fitted_predictor = report.estimator_
    if with_preprocessing:
        fitted_predictor = fitted_predictor.named_steps["predictor"]
    feature_importances = fitted_predictor.feature_importances_
    np.testing.assert_allclose(df["importance"].to_numpy(), feature_importances)

    df = display.frame()
    expected_columns = ["feature", "importance"]
    assert df.columns.tolist() == expected_columns
    assert df["feature"].tolist() == columns_names

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_xlabel() == "Mean Decrease in Impurity (MDI)"
    assert display.ax_.get_ylabel() == ""
    estimator_name = display.importances["estimator"][0]
    assert (
        display.figure_.get_suptitle()
        == f"Mean Decrease in Impurity (MDI) of {estimator_name}"
    )


@pytest.mark.parametrize("with_preprocessing", [True, False])
def test_multi_output_regression(
    pyplot,
    regression_multioutput_train_test_split,
    with_preprocessing,
):
    """Check the attributes and default plotting behaviour of the impurity decrease plot
    with multi-output regression data."""
    X_train, X_test, y_train, y_test = regression_multioutput_train_test_split
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    predictor = RandomForestRegressor(random_state=42)
    if with_preprocessing:
        model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])
    else:
        model = predictor

    report = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.impurity_decrease()
    assert isinstance(display, ImpurityDecreaseDisplay)

    expected_columns = ["estimator", "split", "feature", "importance"]
    df = display.importances
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    assert df["estimator"].nunique() == 1

    fitted_predictor = report.estimator_
    if with_preprocessing:
        fitted_predictor = fitted_predictor.named_steps["predictor"]
    feature_importances = fitted_predictor.feature_importances_
    np.testing.assert_allclose(df["importance"].to_numpy(), feature_importances)

    df = display.frame()
    expected_columns = ["feature", "importance"]
    assert df.columns.tolist() == expected_columns
    assert df["feature"].tolist() == columns_names

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_xlabel() == "Mean Decrease in Impurity (MDI)"
    assert display.ax_.get_ylabel() == ""
    estimator_name = display.importances["estimator"][0]
    assert (
        display.figure_.get_suptitle()
        == f"Mean Decrease in Impurity (MDI) of {estimator_name}"
    )
