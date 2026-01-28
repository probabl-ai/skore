import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils._testing import _convert_container

from skore import CoefficientsDisplay, ComparisonReport, EstimatorReport


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

    report_1 = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
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
    for col in ("split", "output", "label"):
        assert df[col].isna().all()
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())

    for report_name, estimator_report in report.reports_.items():  # noqa: B007
        # report_name is used in the pandas.query as an string which is not detected by
        # ruff as a used variable
        fitted_predictor = estimator_report.estimator_
        if with_preprocessing:
            fitted_predictor = fitted_predictor.named_steps["predictor"]
        coef_with_intercept = np.concatenate(
            [fitted_predictor.intercept_[np.newaxis, :], fitted_predictor.coef_.T]
        )

        coef_split = (
            df.query("estimator == @report_name")[["feature", "coefficients"]]
            .set_index("feature")
            .loc[["Intercept"] + columns_names]
            .to_numpy()
        )

        np.testing.assert_allclose(coef_split, coef_with_intercept)

    df = display.frame(sorting_order=None)
    expected_columns = ["estimator", "feature", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert df["feature"].tolist() == (["Intercept"] + columns_names) * len(
        report.reports_
    )
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())
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
    for report_name, ax in zip(report.reports_, display.ax_, strict=True):  # noqa: B007
        # report_name is used in the pandas.query as an string which is not detected by
        # ruff as a used variable
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
    logistic_multiclass_classification_with_train_test,
    fit_intercept,
    with_preprocessing,
):
    """Check the attributes and default plotting behaviour of the coefficients plot with
    multiclass data."""
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

    report_1 = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
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
    for col in ("split", "output"):
        assert df[col].isna().all()
    np.testing.assert_allclose(
        np.unique(df["label"]).astype(y_train.dtype), range(n_classes)
    )
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())

    for report_name, estimator_report in report.reports_.items():  # noqa: B007
        # report_name is used in the pandas.query as an string which is not detected by
        # ruff as a used variable
        fitted_predictor = estimator_report.estimator_
        if with_preprocessing:
            fitted_predictor = fitted_predictor.named_steps["predictor"]
        coef_with_intercept = np.concatenate(
            [fitted_predictor.intercept_[:, np.newaxis], fitted_predictor.coef_], axis=1
        ).ravel()

        # Extract coefficients in the same order: for each label, for each feature
        df_report = df.query("estimator == @report_name")[
            ["feature", "label", "coefficients"]
        ]
        coef_split = []
        for label in range(n_classes):  # noqa: B007
            # label is used in the pandas.query as an string which is not detected by
            # ruff as a used variable
            df_label = df_report.query("label == @label")
            coef_split.extend(
                df_label.set_index("feature")
                .loc[["Intercept"] + columns_names, "coefficients"]
                .to_numpy()
            )

        np.testing.assert_allclose(coef_split, coef_with_intercept)

    df = display.frame(sorting_order=None)
    expected_columns = ["estimator", "feature", "label", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert np.unique(df["label"]).tolist() == np.unique(y_train).tolist()
    assert df["feature"].tolist() == (["Intercept"] + columns_names) * n_classes * len(
        report.reports_
    )
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())
    if not fit_intercept:
        mask = df["feature"] == "Intercept"
        np.testing.assert_allclose(df.loc[mask, "coefficients"], 0)

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == len(report.reports_)
    for report_name, ax in zip(report.reports_, display.ax_, strict=True):  # noqa: B007
        # report_name is used in the pandas.query as an string which is not detected by
        # ruff as a used variable
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_title() == f"estimator = {report_name}"
        assert ax.get_xlabel() == "Magnitude of coefficient"
        assert ax.get_ylabel() == ""
    # For multiclass, auto subplot_by becomes "estimator"
    assert display.figure_.get_suptitle() == "Coefficients by estimator"

    display.plot(subplot_by="estimator")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == len(report.reports_)
    for report_name, ax in zip(report.reports_, display.ax_, strict=True):  # noqa: B007
        # report_name is used in the pandas.query as an string which is not detected by
        # ruff as a used variable
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_title() == f"estimator = {report_name}"
        assert ax.get_xlabel() == "Magnitude of coefficient"
        assert ax.get_ylabel() == ""
    assert display.figure_.get_suptitle() == "Coefficients by estimator"

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
    single output regression data."""
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

    report_1 = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
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
    for col in ("split", "output", "label"):
        assert df[col].isna().all()
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())

    for report_name, estimator_report in report.reports_.items():  # noqa: B007
        # report_name is used in the pandas.query as an string which is not detected by
        # ruff as a used variable
        fitted_predictor = estimator_report.estimator_
        if with_preprocessing:
            fitted_predictor = fitted_predictor.named_steps["predictor"]
        if with_transformed_target:
            fitted_predictor = fitted_predictor.regressor_
        coef_with_intercept = np.concatenate(
            [
                np.atleast_2d(fitted_predictor.intercept_).T,
                np.atleast_2d(fitted_predictor.coef_),
            ],
            axis=1,
        ).ravel()

        coef_split = (
            df.query("estimator == @report_name")[["feature", "coefficients"]]
            .set_index("feature")
            .loc[["Intercept"] + columns_names]
            .to_numpy()
            .ravel()
        )

        np.testing.assert_allclose(coef_split, coef_with_intercept)

    df = display.frame(sorting_order=None)
    expected_columns = ["estimator", "feature", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert df["feature"].tolist() == (["Intercept"] + columns_names) * len(
        report.reports_
    )
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())
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
    for report_name, ax in zip(report.reports_, display.ax_, strict=True):  # noqa: B007
        # report_name is used in the pandas.query as an string which is not detected by
        # ruff as a used variable
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
    linear_regression_multioutput_with_train_test,
    fit_intercept,
    with_preprocessing,
    with_transformed_target,
):
    """Check the attributes and default plotting behaviour of the coefficients plot with
    multi-output regression data."""
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

    report_1 = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
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
    for col in ("split", "label"):
        assert df[col].isna().all()
    assert np.unique(df["output"]).tolist() == [f"{i}" for i in range(n_outputs)]
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())

    for report_name, estimator_report in report.reports_.items():  # noqa: B007
        # report_name is used in the pandas.query as an string which is not detected by
        # ruff as a used variable
        fitted_predictor = estimator_report.estimator_
        if with_preprocessing:
            fitted_predictor = fitted_predictor.named_steps["predictor"]
        if with_transformed_target:
            fitted_predictor = fitted_predictor.regressor_

        if fit_intercept:
            intercept = np.atleast_2d(fitted_predictor.intercept_).T
        else:
            intercept = np.zeros((n_outputs, 1))

        coef_with_intercept = np.concatenate(
            [intercept, fitted_predictor.coef_], axis=1
        ).ravel()

        df_report = df.query("estimator == @report_name")[
            ["feature", "output", "coefficients"]
        ]
        coef_split = []
        for output in range(n_outputs):
            df_output = df_report.query(f"output == '{output}'")
            coef_split.extend(
                df_output.set_index("feature")
                .loc[["Intercept"] + columns_names, "coefficients"]
                .to_numpy()
            )

        np.testing.assert_allclose(coef_split, coef_with_intercept)

    df = display.frame(sorting_order=None)
    expected_columns = ["estimator", "feature", "output", "coefficients"]
    assert df.columns.tolist() == expected_columns
    assert np.unique(df["output"]).tolist() == [f"{i}" for i in range(n_outputs)]
    assert df["feature"].tolist() == (["Intercept"] + columns_names) * n_outputs * len(
        report.reports_
    )
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())
    if not fit_intercept:
        mask = df["feature"] == "Intercept"
        np.testing.assert_allclose(df.loc[mask, "coefficients"], 0)

    display.plot()
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    # For multi-output regression with comparison reports, plot() creates subplots
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == len(report.reports_)
    for report_name, ax in zip(report.reports_, display.ax_, strict=True):  # noqa: B007
        # report_name is used in the pandas.query as an string which is not detected by
        # ruff as a used variable
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_title() == f"estimator = {report_name}"
        assert ax.get_xlabel() == "Magnitude of coefficient"
        assert ax.get_ylabel() == ""
    # For multi-output regression, auto subplot_by becomes "estimator"
    assert display.figure_.get_suptitle() == "Coefficients by estimator"

    display.plot(subplot_by="estimator")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == len(report.reports_)
    for report_name, ax in zip(report.reports_, display.ax_, strict=True):  # noqa: B007
        # report_name is used in the pandas.query as an string which is not detected by
        # ruff as a used variable
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_title() == f"estimator = {report_name}"
        assert ax.get_xlabel() == "Magnitude of coefficient"
        assert ax.get_ylabel() == ""
    assert display.figure_.get_suptitle() == "Coefficients by estimator"

    with pytest.raises(ValueError, match="Column incorrect not found in the frame"):
        display.plot(subplot_by="incorrect")


@pytest.mark.parametrize(
    "fixture_name",
    [
        "logistic_multiclass_classification_with_train_test",
        "linear_regression_multioutput_with_train_test",
    ],
)
def test_subplot_by_none_multiclass_or_multioutput(
    pyplot,
    request,
    fixture_name,
):
    """Check that an error is raised when `subplot_by=None` and there are multiple
    labels (multiclass) or outputs (multi-output regression)."""
    fixture = request.getfixturevalue(fixture_name)
    estimator, X_train, X_test, y_train, y_test = fixture
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    report_1 = EstimatorReport(
        clone(estimator), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        clone(estimator), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

    display = report.feature_importance.coefficients()

    err_msg = (
        "There are multiple labels or outputs and `subplot_by` is `None`. "
        "There is too much information to display on a single plot. "
        "Please provide a column to group by using `subplot_by`."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by=None)


def test_different_features(
    pyplot,
    logistic_multiclass_classification_with_train_test,
):
    """Check that we get a proper report even if the estimators do not have the same
    input features."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)
    n_classes = len(np.unique(y_train))

    simple_model = clone(estimator)
    complex_model = Pipeline(
        [("poly", PolynomialFeatures()), ("predictor", clone(estimator))]
    )

    report_simple = EstimatorReport(
        simple_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_complex = EstimatorReport(
        complex_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"report_simple": report_simple, "report_complex": report_complex}
    )

    display = report.feature_importance.coefficients()
    assert isinstance(display, CoefficientsDisplay)

    df = display.frame(sorting_order=None)
    expected_features = [
        "Intercept"
    ] + report_simple.estimator_.feature_names_in_.tolist()
    assert (
        df.query("estimator == 'report_simple'")["feature"].tolist()
        == expected_features * n_classes
    )

    expected_features = ["Intercept"] + report_complex.estimator_[
        :-1
    ].get_feature_names_out().tolist()
    assert (
        df.query("estimator == 'report_complex'")["feature"].tolist()
        == expected_features * n_classes
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
        assert display.figure_.get_suptitle() == "Coefficients by estimator"


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

    report_1 = EstimatorReport(
        clone(estimator), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        clone(estimator), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

    display = report.feature_importance.coefficients()

    assert display.frame(include_intercept=False).query("feature == 'Intercept'").empty

    display.plot(include_intercept=False)
    assert all(
        label.get_text() != "Intercept" for label in display.ax_.get_yticklabels()
    )
    assert display.figure_.get_suptitle() == "Coefficients"
