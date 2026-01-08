import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils._testing import _convert_container

from skore import CoefficientsDisplay, ComparisonReport, CrossValidationReport


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

    report_1 = CrossValidationReport(model, X, y, splitter=splitter)
    report_2 = CrossValidationReport(model, X, y, splitter=splitter)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

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
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())

    for report_name, cv_report in report.reports_.items():  # noqa: B007
        for split_index, estimator_report in enumerate(cv_report.estimator_reports_):  # noqa: B007
            # split_index is used in the pandas.query as an string which is not detected
            # by ruff as a used variable
            fitted_predictor = estimator_report.estimator_
            if with_preprocessing:
                fitted_predictor = fitted_predictor.named_steps["predictor"]
            coef_with_intercept = np.concatenate(
                [fitted_predictor.intercept_[np.newaxis, :], fitted_predictor.coef_.T]
            ).ravel()

            coef_split = (
                df.query("estimator == @report_name & split == @split_index")
                .loc[:, ["feature", "coefficients"]]
                .set_index("feature")
                .loc[["Intercept"] + columns_names]
                .to_numpy()
                .ravel()
            )

            np.testing.assert_allclose(coef_split, coef_with_intercept)

    df = display.frame()
    expected_columns = ["estimator", "split", "feature", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert df["feature"].tolist() == (["Intercept"] + columns_names) * splitter * len(
        report.reports_
    )
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())
    assert df["split"].nunique() == splitter
    if not fit_intercept:
        mask = df["feature"] == "Intercept"
        np.testing.assert_allclose(df.loc[mask, "coefficients"], 0)

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_xlabel() == "Magnitude of coefficient"
    assert display.ax_.get_ylabel() == ""

    display.plot(subplot_by="estimator")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == len(report.reports_)
    for report_name, ax in zip(report.reports_, display.ax_, strict=True):
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_title() == f"estimator = {report_name}"
        assert ax.get_xlabel() == "Magnitude of coefficient"
        assert ax.get_ylabel() == ""

    with pytest.raises(ValueError, match="Column incorrect not found in the frame"):
        display.plot(subplot_by="incorrect")


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("with_preprocessing", [True, False])
def test_multiclass_classification(
    pyplot,
    logistic_multiclass_classification_data,
    fit_intercept,
    with_preprocessing,
):
    """Check the attributes and default plotting behaviour of the coefficients plot with
    multiclass data."""
    estimator, X, y = logistic_multiclass_classification_data
    columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
    X = _convert_container(X, "dataframe", columns_name=columns_names)
    n_classes = len(np.unique(y))
    splitter = 2

    predictor = clone(estimator).set_params(fit_intercept=fit_intercept)
    if with_preprocessing:
        model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])
    else:
        model = predictor

    report_1 = CrossValidationReport(model, X, y, splitter=splitter)
    report_2 = CrossValidationReport(model, X, y, splitter=splitter)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

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
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())

    for report_name, cv_report in report.reports_.items():  # noqa: B007
        for split_index, estimator_report in enumerate(cv_report.estimator_reports_):  # noqa: B007
            # split_index is used in the pandas.query as an string which is not detected
            # by ruff as a used variable
            fitted_predictor = estimator_report.estimator_
            if with_preprocessing:
                fitted_predictor = fitted_predictor.named_steps["predictor"]
            coef_with_intercept = np.concatenate(
                [fitted_predictor.intercept_[np.newaxis, :], fitted_predictor.coef_.T]
            ).ravel()

            coef_split = (
                df.query("estimator == @report_name & split == @split_index")
                .loc[:, ["feature", "label", "coefficients"]]
                .pivot(index="feature", columns="label", values="coefficients")
                .loc[["Intercept"] + columns_names]
                .to_numpy()
                .ravel()
            )

            np.testing.assert_allclose(coef_split, coef_with_intercept)

    df = display.frame()
    expected_columns = ["estimator", "split", "feature", "label", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert np.unique(df["label"]).tolist() == np.unique(y).tolist()
    assert df["feature"].tolist() == (
        (["Intercept"] + columns_names) * n_classes * splitter * len(report.reports_)
    )
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())
    assert df["split"].nunique() == splitter
    if not fit_intercept:
        mask = df["feature"] == "Intercept"
        np.testing.assert_allclose(df.loc[mask, "coefficients"], 0)

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == len(report.reports_)
    for report_name, ax in zip(report.reports_, display.ax_, strict=True):
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_title() == f"estimator = {report_name}"
        assert ax.get_xlabel() == "Magnitude of coefficient"
        assert ax.get_ylabel() == ""

    display.plot(subplot_by="estimator")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == len(report.reports_)
    for report_name, ax in zip(report.reports_, display.ax_, strict=True):
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_title() == f"estimator = {report_name}"
        assert ax.get_xlabel() == "Magnitude of coefficient"
        assert ax.get_ylabel() == ""

    with pytest.raises(ValueError, match="Column incorrect not found in the frame"):
        display.plot(subplot_by="incorrect")


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
    single output regression data."""
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

    report_1 = CrossValidationReport(model, X, y, splitter=splitter)
    report_2 = CrossValidationReport(model, X, y, splitter=splitter)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

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
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())

    for report_name, cv_report in report.reports_.items():  # noqa: B007
        for split_index, estimator_report in enumerate(cv_report.estimator_reports_):  # noqa: B007
            # split_index is used in the pandas.query as an string which is not detected
            # by ruff as a used variable
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
            ).ravel()

            coef_split = (
                df.query("estimator == @report_name & split == @split_index")
                .loc[:, ["feature", "coefficients"]]
                .set_index("feature")
                .loc[["Intercept"] + columns_names]
                .to_numpy()
                .ravel()
            )

            np.testing.assert_allclose(coef_split, coef_with_intercept)

    df = display.frame()
    expected_columns = ["estimator", "split", "feature", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert df["feature"].tolist() == (["Intercept"] + columns_names) * splitter * len(
        report.reports_
    )
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())
    assert df["split"].nunique() == splitter
    if not fit_intercept:
        mask = df["feature"] == "Intercept"
        np.testing.assert_allclose(df.loc[mask, "coefficients"], 0)

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_xlabel() == "Magnitude of coefficient"
    assert display.ax_.get_ylabel() == ""

    display.plot(subplot_by="estimator")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == len(report.reports_)
    for report_name, ax in zip(report.reports_, display.ax_, strict=True):
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_title() == f"estimator = {report_name}"
        assert ax.get_xlabel() == "Magnitude of coefficient"
        assert ax.get_ylabel() == ""

    with pytest.raises(ValueError, match="Column incorrect not found in the frame"):
        display.plot(subplot_by="incorrect")


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
    multi-output regression data."""
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

    report_1 = CrossValidationReport(model, X, y, splitter=splitter)
    report_2 = CrossValidationReport(model, X, y, splitter=splitter)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

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
    assert df["label"].isna().all()
    assert np.unique(df["output"]).tolist() == [f"{i}" for i in range(n_outputs)]
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())

    for report_name, cv_report in report.reports_.items():  # noqa: B007
        for split_index, estimator_report in enumerate(cv_report.estimator_reports_):  # noqa: B007
            # split_index is used in the pandas.query as an string which is not detected
            # by ruff as a used variable
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
            coef_with_intercept = np.concatenate(
                [intercept, fitted_predictor.coef_.T]
            ).ravel()

            coef_split = (
                df.query("estimator == @report_name & split == @split_index")
                .loc[:, ["feature", "output", "coefficients"]]
                .pivot(index="feature", columns="output", values="coefficients")
                .loc[["Intercept"] + columns_names]
                .to_numpy()
                .ravel()
            )

            np.testing.assert_allclose(coef_split, coef_with_intercept)

    df = display.frame()
    expected_columns = ["estimator", "split", "feature", "output", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert np.unique(df["output"]).tolist() == [f"{i}" for i in range(n_outputs)]
    assert df["feature"].tolist() == (
        ["Intercept"] + columns_names
    ) * n_outputs * splitter * len(report.reports_)
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())
    assert df["split"].nunique() == splitter
    if not fit_intercept:
        mask = df["feature"] == "Intercept"
        np.testing.assert_allclose(df.loc[mask, "coefficients"], 0)

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == len(report.reports_)
    for report_name, ax in zip(report.reports_, display.ax_, strict=True):
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_title() == f"estimator = {report_name}"
        assert ax.get_xlabel() == "Magnitude of coefficient"
        assert ax.get_ylabel() == ""

    display.plot(subplot_by="estimator")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == len(report.reports_)
    for report_name, ax in zip(report.reports_, display.ax_, strict=True):
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_title() == f"estimator = {report_name}"
        assert ax.get_xlabel() == "Magnitude of coefficient"
        assert ax.get_ylabel() == ""

    with pytest.raises(ValueError, match="Column incorrect not found in the frame"):
        display.plot(subplot_by="incorrect")


def test_different_features(
    pyplot,
    logistic_multiclass_classification_data,
):
    """Check that we get a proper report even if the estimators do not have the same
    input features."""
    estimator, X, y = logistic_multiclass_classification_data
    columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
    X = _convert_container(X, "dataframe", columns_name=columns_names)
    n_classes, splitter = len(np.unique(y)), 2

    simple_model = clone(estimator)
    complex_model = Pipeline(
        [("poly", PolynomialFeatures()), ("predictor", clone(estimator))]
    )

    report_simple = CrossValidationReport(simple_model, X, y, splitter=splitter)
    report_complex = CrossValidationReport(complex_model, X, y, splitter=splitter)
    report = ComparisonReport(
        reports={"report_simple": report_simple, "report_complex": report_complex}
    )

    display = report.feature_importance.coefficients()
    assert isinstance(display, CoefficientsDisplay)

    df = display.frame()
    expected_features = ["Intercept"] + report_simple.estimator_reports_[
        0
    ].estimator_.feature_names_in_.tolist()
    assert (
        df.query("estimator == 'report_simple'")["feature"].tolist()
        == expected_features * n_classes * splitter
    )

    expected_features = ["Intercept"] + report_complex.estimator_reports_[0].estimator_[
        :-1
    ].get_feature_names_out().tolist()
    assert (
        df.query("estimator == 'report_complex'")["feature"].tolist()
        == expected_features * n_classes * splitter
    )

    err_msg = (
        "The estimators have different features and should be plotted on different "
        "axis using `subplot_by='estimator'`."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="label")

    for subplot_by in ("auto", "estimator"):
        display.plot(subplot_by=subplot_by)
        assert hasattr(display, "figure_")
        assert hasattr(display, "ax_")
        assert isinstance(display.ax_, np.ndarray)
        assert len(display.ax_) == len(report.reports_)
        for report_name, ax in zip(report.reports_, display.ax_, strict=True):
            assert isinstance(ax, mpl.axes.Axes)
            assert ax.get_title() == f"estimator = {report_name}"
            assert ax.get_xlabel() == "Magnitude of coefficient"
            assert ax.get_ylabel() == ""


def test_include_intercept(
    pyplot,
    logistic_binary_classification_data,
):
    """Check whether or not we can include or exclude the intercept."""
    estimator, X, y = logistic_binary_classification_data
    columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
    X = _convert_container(X, "dataframe", columns_name=columns_names)
    splitter = 2

    report_1 = CrossValidationReport(clone(estimator), X, y, splitter=splitter)
    report_2 = CrossValidationReport(clone(estimator), X, y, splitter=splitter)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

    display = report.feature_importance.coefficients()

    assert display.frame(include_intercept=False).query("feature == 'Intercept'").empty

    display.plot(include_intercept=False)
    assert all(
        label.get_text() != "Intercept" for label in display.ax_.get_yticklabels()
    )
