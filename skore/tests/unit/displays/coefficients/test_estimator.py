import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import _convert_container

from skore import CoefficientsDisplay, EstimatorReport


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("with_preprocessing", [True, False])
def test_binary_classification(
    pyplot,
    logistic_binary_classification_with_train_test,
    fit_intercept,
    with_preprocessing,
):
    """Check the attributes and default plotting behaviour of the coefficients plot with
    binary data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    predictor = clone(estimator).set_params(fit_intercept=fit_intercept)
    if with_preprocessing:
        model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])
    else:
        model = predictor

    report = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.coefficients()
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
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    for col in ("split", "output", "label"):
        assert df[col].isna().all()
    assert df["estimator"].nunique() == 1

    fitted_predictor = report.estimator_
    if with_preprocessing:
        fitted_predictor = fitted_predictor.named_steps["predictor"]
    coef = np.concatenate(
        [fitted_predictor.intercept_[:, np.newaxis], fitted_predictor.coef_], axis=1
    ).ravel()
    np.testing.assert_allclose(df["coefficients"].to_numpy(), coef)

    df = display.frame(sorting_order=None)
    expected_columns = ["feature", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert df["feature"].tolist() == ["Intercept"] + columns_names
    if not fit_intercept:
        mask = df["feature"] == "Intercept"
        assert df.loc[mask, "coefficients"].item() == pytest.approx(0)

    result = display.plot()
    assert isinstance(result.axes[0, 0], mpl.axes.Axes)

    assert result.axes[0, 0].get_xlabel() == "Magnitude of coefficient"
    assert result.axes[0, 0].get_ylabel() == ""
    estimator_name = display.coefficients["estimator"][0]
    assert result.figure.get_suptitle() == f"Coefficients of {estimator_name}"

    with pytest.raises(ValueError, match="No columns to group by."):
        display.plot(subplot_by="label")


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("with_preprocessing", [True, False])
def test_multiclass_classification(
    pyplot,
    logistic_multiclass_classification_with_train_test,
    fit_intercept,
    with_preprocessing,
):
    """Check the attributes and default plotting behaviour of the coefficients plot with
    binary data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)
    n_classes = len(np.unique(y_train))

    predictor = clone(estimator).set_params(fit_intercept=fit_intercept)
    if with_preprocessing:
        model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])
    else:
        model = predictor

    report = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.coefficients()
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
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    for col in ("split", "output"):
        assert df[col].isna().all()
    np.testing.assert_allclose(
        np.unique(df["label"]).astype(y_train.dtype), range(n_classes)
    )
    assert df["estimator"].nunique() == 1

    fitted_predictor = report.estimator_
    if with_preprocessing:
        fitted_predictor = fitted_predictor.named_steps["predictor"]
    coef = np.concatenate(
        [fitted_predictor.intercept_[:, np.newaxis], fitted_predictor.coef_], axis=1
    ).ravel()
    np.testing.assert_allclose(df["coefficients"].to_numpy(), coef)

    df = display.frame(sorting_order=None)
    expected_columns = ["feature", "label", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert np.unique(df["label"]).tolist() == np.unique(y_train).tolist()
    assert df["feature"].tolist() == (["Intercept"] + columns_names) * n_classes
    if not fit_intercept:
        mask = df["feature"] == "Intercept"
        np.testing.assert_allclose(df.loc[mask, "coefficients"], 0)

    result = display.plot()
    assert isinstance(result.axes[0, 0], mpl.axes.Axes)

    assert result.axes[0, 0].get_xlabel() == "Magnitude of coefficient"
    assert result.axes[0, 0].get_ylabel() == ""
    estimator_name = display.coefficients["estimator"][0]
    assert result.figure.get_suptitle() == f"Coefficients of {estimator_name}"

    result = display.plot(subplot_by="label")
    assert isinstance(result.axes.flatten(), np.ndarray)
    assert len(result.axes.flatten()) == n_classes
    for label, ax in enumerate(result.axes.flatten()):
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_title() == f"label = {label}"
        assert ax.get_xlabel() == "Magnitude of coefficient"
        assert ax.get_ylabel() == ""
    assert result.figure.get_suptitle() == f"Coefficients of {estimator_name} by label"

    with pytest.raises(ValueError, match="Column incorrect not found in the frame"):
        display.plot(subplot_by="incorrect")


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("with_preprocessing", [True, False])
@pytest.mark.parametrize("with_transformed_target", [True, False])
def test_single_output_regression(
    pyplot,
    linear_regression_with_train_test,
    fit_intercept,
    with_preprocessing,
    with_transformed_target,
):
    """Check the attributes and default plotting behaviour of the coefficients plot with
    binary data."""
    estimator, X_train, X_test, y_train, y_test = linear_regression_with_train_test
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    predictor = clone(estimator).set_params(fit_intercept=fit_intercept)
    if with_transformed_target:
        predictor = TransformedTargetRegressor(predictor)
    if with_preprocessing:
        model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])
    else:
        model = predictor

    report = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.coefficients()
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
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    for col in ("split", "output", "label"):
        assert df[col].isna().all()
    assert df["estimator"].nunique() == 1

    fitted_predictor = report.estimator_
    if with_preprocessing:
        fitted_predictor = fitted_predictor.named_steps["predictor"]
    if with_transformed_target:
        fitted_predictor = fitted_predictor.regressor_
    coef = np.concatenate(
        [
            np.atleast_2d(fitted_predictor.intercept_).T,
            np.atleast_2d(fitted_predictor.coef_),
        ],
        axis=1,
    ).ravel()
    np.testing.assert_allclose(df["coefficients"].to_numpy(), coef)

    df = display.frame(sorting_order=None)
    expected_columns = ["feature", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert df["feature"].tolist() == ["Intercept"] + columns_names
    if not fit_intercept:
        mask = df["feature"] == "Intercept"
        assert df.loc[mask, "coefficients"].item() == pytest.approx(0)

    result = display.plot()
    assert isinstance(result.axes[0, 0], mpl.axes.Axes)

    assert result.axes[0, 0].get_xlabel() == "Magnitude of coefficient"
    assert result.axes[0, 0].get_ylabel() == ""
    estimator_name = display.coefficients["estimator"][0]
    assert result.figure.get_suptitle() == f"Coefficients of {estimator_name}"

    with pytest.raises(ValueError, match="No columns to group by."):
        display.plot(subplot_by="output")


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("with_preprocessing", [True, False])
@pytest.mark.parametrize("with_transformed_target", [True, False])
def test_multi_output_regression(
    pyplot,
    linear_regression_multioutput_with_train_test,
    fit_intercept,
    with_preprocessing,
    with_transformed_target,
):
    """Check the attributes and default plotting behaviour of the coefficients plot with
    binary data."""
    estimator, X_train, X_test, y_train, y_test = (
        linear_regression_multioutput_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)
    n_outputs = y_train.shape[1]

    predictor = clone(estimator).set_params(fit_intercept=fit_intercept)
    if with_transformed_target:
        predictor = TransformedTargetRegressor(predictor)
    if with_preprocessing:
        model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])
    else:
        model = predictor

    report = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.coefficients()
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
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    for col in ("split", "label"):
        assert df[col].isna().all()
    assert np.unique(df["output"]).tolist() == [f"{i}" for i in range(n_outputs)]
    assert df["estimator"].nunique() == 1

    fitted_predictor = report.estimator_
    if with_preprocessing:
        fitted_predictor = fitted_predictor.named_steps["predictor"]
    if with_transformed_target:
        fitted_predictor = fitted_predictor.regressor_

    if fit_intercept:
        intercept = np.atleast_2d(fitted_predictor.intercept_).T
    else:
        intercept = np.zeros((n_outputs, 1))

    coef = np.concatenate([intercept, fitted_predictor.coef_], axis=1).ravel()
    np.testing.assert_allclose(df["coefficients"].to_numpy(), coef)

    df = display.frame(sorting_order=None)
    expected_columns = ["feature", "output", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert np.unique(df["output"]).tolist() == [f"{i}" for i in range(n_outputs)]
    assert df["feature"].tolist() == (["Intercept"] + columns_names) * n_outputs
    if not fit_intercept:
        mask = df["feature"] == "Intercept"
        np.testing.assert_allclose(df.loc[mask, "coefficients"], 0)

    result = display.plot()
    assert isinstance(result.axes[0, 0], mpl.axes.Axes)

    assert result.axes[0, 0].get_xlabel() == "Magnitude of coefficient"
    assert result.axes[0, 0].get_ylabel() == ""
    estimator_name = display.coefficients["estimator"][0]
    assert result.figure.get_suptitle() == f"Coefficients of {estimator_name}"

    result = display.plot(subplot_by="output")
    assert isinstance(result.axes.flatten(), np.ndarray)
    assert len(result.axes.flatten()) == n_outputs
    for output, ax in enumerate(result.axes.flatten()):
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_title() == f"output = {output}"
        assert ax.get_xlabel() == "Magnitude of coefficient"
        assert ax.get_ylabel() == ""
    assert result.figure.get_suptitle() == f"Coefficients of {estimator_name} by output"

    with pytest.raises(ValueError, match="Column incorrect not found in the frame"):
        display.plot(subplot_by="incorrect")


def test_include_intercept(
    pyplot,
    logistic_binary_classification_with_train_test,
):
    """Check whether or not we can include or exclude the intercept."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    report = EstimatorReport(
        clone(estimator), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.coefficients()

    assert display.frame(include_intercept=False).query("feature == 'Intercept'").empty

    result = display.plot(include_intercept=False)
    assert all(
        label.get_text() != "Intercept" for label in result.axes[0, 0].get_yticklabels()
    )
    estimator_name = display.coefficients["estimator"][0]
    assert result.figure.get_suptitle() == f"Coefficients of {estimator_name}"
