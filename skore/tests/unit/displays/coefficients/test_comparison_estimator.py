import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import _convert_container

from skore import ComparisonReport, EstimatorReport
from skore._sklearn._plot.feature_importance.coefficients import CoefficientsDisplay


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

    df = display.frame()
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

    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == "Estimator"
    assert [t.get_text() for t in legend.get_texts()] == [
        f"{report_name}" for report_name in list(report.reports_.keys())
    ]

    display.plot(subplots_by="estimator")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == len(report.reports_)
    for report_name, ax in zip(report.reports_, display.ax_, strict=True):  # noqa: B007
        # report_name is used in the pandas.query as an string which is not detected by
        # ruff as a used variable
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_title() == f"Estimator: {report_name}"
        assert ax.get_xlabel() == "Magnitude of coefficient"
        assert ax.get_ylabel() == ""
        assert ax.get_legend() is None
