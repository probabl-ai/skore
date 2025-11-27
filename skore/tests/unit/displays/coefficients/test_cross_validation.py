import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import _convert_container

from skore import CrossValidationReport
from skore._sklearn._plot.feature_importance.coefficients import CoefficientsDisplay


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("with_preprocessing", [True, False])
def test_binary_classification(
    pyplot,
    logistic_binary_classification_data,
    fit_intercept,
    with_preprocessing,
):
    """Check the attributes and default plotting behaviour of the coefficients plot with
    binary data."""
    estimator, X, y = logistic_binary_classification_data
    columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
    X = _convert_container(X, "dataframe", columns_name=columns_names)
    splitter = 2

    predictor = clone(estimator).set_params(fit_intercept=fit_intercept)
    if with_preprocessing:
        model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])
    else:
        model = predictor

    report = CrossValidationReport(model, X, y, splitter=splitter)

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
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    assert df["split"].nunique() == splitter
    for col in ("output", "label"):
        assert df[col].isna().all()
    assert df["estimator"].nunique() == 1

    for split_index, estimator_report in enumerate(report.estimator_reports_):  # noqa: B007
        # split_index is used in the pandas.query as an string which is not detected by
        # ruff as a used variable
        fitted_predictor = estimator_report.estimator_
        if with_preprocessing:
            fitted_predictor = fitted_predictor.named_steps["predictor"]
        coef_with_intercept = np.concatenate(
            [fitted_predictor.intercept_[np.newaxis, :], fitted_predictor.coef_.T]
        )

        coef_split = (
            df.query("split == @split_index")[["feature", "coefficients"]]
            .set_index("feature")
            .loc[["Intercept"] + columns_names]
            .to_numpy()
        )

        np.testing.assert_allclose(coef_split, coef_with_intercept)

    df = display.frame()
    expected_columns = ["split", "feature", "coefficients"]
    assert df.columns.tolist() == expected_columns
    feature_names = (["Intercept"] + columns_names) * splitter
    assert df["feature"].tolist() == feature_names
    assert df["split"].nunique() == splitter
    if not fit_intercept:
        mask = df["feature"] == "Intercept"
        np.testing.assert_allclose(df.loc[mask, "coefficients"], 0)

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_legend() is None
    assert display.ax_.get_title() == report.estimator_name_
    assert display.ax_.get_xlabel() == "Magnitude of coefficient"
    assert display.ax_.get_ylabel() == ""


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("with_preprocessing", [True, False])
def test_multiclass_classification(
    pyplot,
    logistic_multiclass_classification_data,
    fit_intercept,
    with_preprocessing,
):
    """Check the attributes and default plotting behaviour of the coefficients plot with
    binary data."""
    estimator, X, y = logistic_multiclass_classification_data
    columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
    X = _convert_container(X, "dataframe", columns_name=columns_names)
    n_classes, splitter = len(np.unique(y)), 2

    predictor = clone(estimator).set_params(fit_intercept=fit_intercept)
    if with_preprocessing:
        model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])
    else:
        model = predictor

    report = CrossValidationReport(model, X, y, splitter=splitter)

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
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    assert df["split"].nunique() == splitter
    assert df["output"].isna().all()
    np.testing.assert_allclose(np.unique(df["label"]).astype(y.dtype), range(n_classes))
    assert df["estimator"].nunique() == 1

    for split_index, estimator_report in enumerate(report.estimator_reports_):  # noqa: B007
        # split_index is used in the pandas.query as an string which is not detected by
        # ruff as a used variable
        fitted_predictor = estimator_report.estimator_
        if with_preprocessing:
            fitted_predictor = fitted_predictor.named_steps["predictor"]
        coef_with_intercept = np.concatenate(
            [fitted_predictor.intercept_[np.newaxis, :], fitted_predictor.coef_.T]
        )

        coef_split = (
            df.query("split == @split_index")[["feature", "label", "coefficients"]]
            .pivot(index="feature", columns="label", values="coefficients")
            .loc[["Intercept"] + columns_names]
            .to_numpy()
        )

        np.testing.assert_allclose(coef_split, coef_with_intercept)

    df = display.frame()
    expected_columns = ["split", "feature", "label", "coefficients"]
    assert df.columns.tolist() == expected_columns
    feature_names = (["Intercept"] + columns_names) * splitter * n_classes
    assert df["feature"].tolist() == feature_names
    assert df["split"].nunique() == splitter
    if not fit_intercept:
        mask = df["feature"] == "Intercept"
        np.testing.assert_allclose(df.loc[mask, "coefficients"], 0)

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_title() == report.estimator_name_
    assert display.ax_.get_xlabel() == "Magnitude of coefficient"
    assert display.ax_.get_ylabel() == ""

    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == "Label"
    assert [t.get_text() for t in legend.get_texts()] == [
        f"{i}" for i in range(n_classes)
    ]


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("with_preprocessing", [True, False])
@pytest.mark.parametrize("with_transformed_target", [True, False])
def test_single_output_regression(
    pyplot,
    linear_regression_data,
    fit_intercept,
    with_preprocessing,
    with_transformed_target,
):
    """Check the attributes and default plotting behaviour of the coefficients plot with
    binary data."""
    estimator, X, y = linear_regression_data
    columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
    X = _convert_container(X, "dataframe", columns_name=columns_names)
    splitter = 2

    predictor = clone(estimator).set_params(fit_intercept=fit_intercept)
    if with_transformed_target:
        predictor = TransformedTargetRegressor(predictor)
    if with_preprocessing:
        model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])
    else:
        model = predictor

    report = CrossValidationReport(model, X, y, splitter=splitter)

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
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    for col in ("output", "label"):
        assert df[col].isna().all()
    assert df["estimator"].nunique() == 1

    for split_index, estimator_report in enumerate(report.estimator_reports_):  # noqa: B007
        # split_index is used in the pandas.query as an string which is not detected by
        # ruff as a used variable
        fitted_predictor = estimator_report.estimator_
        if with_preprocessing:
            fitted_predictor = fitted_predictor.named_steps["predictor"]
        if with_transformed_target:
            fitted_predictor = fitted_predictor.regressor_

        coef_with_intercept = np.concatenate(
            [
                np.atleast_2d(fitted_predictor.intercept_).T,
                np.atleast_2d(fitted_predictor.coef_).T,
            ]
        )

        coef_split = (
            df.query("split == @split_index")[["feature", "coefficients"]]
            .set_index("feature")
            .loc[["Intercept"] + columns_names]
            .to_numpy()
        )

        np.testing.assert_allclose(coef_split, coef_with_intercept)

    df = display.frame()
    expected_columns = ["split", "feature", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert df["feature"].tolist() == (["Intercept"] + columns_names) * splitter
    assert df["split"].nunique() == splitter
    if not fit_intercept:
        mask = df["feature"] == "Intercept"
        np.testing.assert_allclose(df.loc[mask, "coefficients"], 0)

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_legend() is None
    assert display.ax_.get_title() == report.estimator_name_
    assert display.ax_.get_xlabel() == "Magnitude of coefficient"
    assert display.ax_.get_ylabel() == ""


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("with_preprocessing", [True, False])
@pytest.mark.parametrize("with_transformed_target", [True, False])
def test_multi_output_regression(
    pyplot,
    linear_regression_multioutput_data,
    fit_intercept,
    with_preprocessing,
    with_transformed_target,
):
    """Check the attributes and default plotting behaviour of the coefficients plot with
    binary data."""
    estimator, X, y = linear_regression_multioutput_data
    columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
    X = _convert_container(X, "dataframe", columns_name=columns_names)
    n_outputs = y.shape[1]
    splitter = 2

    predictor = clone(estimator).set_params(fit_intercept=fit_intercept)
    if with_transformed_target:
        predictor = TransformedTargetRegressor(predictor)
    if with_preprocessing:
        model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])
    else:
        model = predictor

    report = CrossValidationReport(model, X, y, splitter=splitter)

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
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    assert df["label"].isna().all()
    assert np.unique(df["output"]).tolist() == [f"{i}" for i in range(n_outputs)]
    assert df["estimator"].nunique() == 1

    for split_index, estimator_report in enumerate(report.estimator_reports_):  # noqa: B007
        # split_index is used in the pandas.query as an string which is not detected by
        # ruff as a used variable
        fitted_predictor = estimator_report.estimator_
        if with_preprocessing:
            fitted_predictor = fitted_predictor.named_steps["predictor"]
        if with_transformed_target:
            fitted_predictor = fitted_predictor.regressor_

        if isinstance(fitted_predictor.intercept_, float):
            intercept = np.repeat(fitted_predictor.intercept_, n_outputs, axis=0)[
                np.newaxis, :
            ]
        else:
            intercept = fitted_predictor.intercept_[np.newaxis, :]
        coef_with_intercept = np.concatenate([intercept, fitted_predictor.coef_.T])

        coef_split = (
            df.query("split == @split_index")[["feature", "output", "coefficients"]]
            .pivot(index="feature", columns="output", values="coefficients")
            .loc[["Intercept"] + columns_names]
            .to_numpy()
        )

        np.testing.assert_allclose(coef_split, coef_with_intercept)

    df = display.frame()
    expected_columns = ["split", "feature", "output", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert np.unique(df["output"]).tolist() == [f"{i}" for i in range(n_outputs)]
    assert (
        df["feature"].tolist() == (["Intercept"] + columns_names) * n_outputs * splitter
    )
    assert df["split"].nunique() == splitter
    if not fit_intercept:
        mask = df["feature"] == "Intercept"
        np.testing.assert_allclose(df.loc[mask, "coefficients"], 0)

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_title() == report.estimator_name_
    assert display.ax_.get_xlabel() == "Magnitude of coefficient"
    assert display.ax_.get_ylabel() == ""

    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == "Output"
    assert [t.get_text() for t in legend.get_texts()] == [
        f"{i}" for i in range(n_outputs)
    ]
