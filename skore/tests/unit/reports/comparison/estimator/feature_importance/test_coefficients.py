import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from skore import CoefficientsDisplay, ComparisonReport, EstimatorReport


@pytest.mark.parametrize(
    "estimator",
    [
        Ridge(),
        TransformedTargetRegressor(Ridge()),
        Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("transformed_ridge", TransformedTargetRegressor(Ridge())),
            ]
        ),
    ],
)
def test_with_model_exposing_coef(regression_train_test_split, estimator):
    """Check that we can create a coefficients display from model exposing a `coef_`
    attribute."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert hasattr(report.feature_importance, "coefficients")
    display = report.feature_importance.coefficients()
    assert isinstance(display, CoefficientsDisplay)


def test_with_model_not_exposing_coef(regression_train_test_split):
    """Check that we cannot create a coefficients display from model not exposing a
    `coef_` attribute."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        DecisionTreeRegressor(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        DecisionTreeRegressor(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert not hasattr(report.feature_importance, "coefficients")


def test_with_mixed_reports(regression_train_test_split):
    """Check that we cannot create a coefficients display from mixed reports."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        Ridge(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        DecisionTreeRegressor(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert not hasattr(report.feature_importance, "coefficients")


def test_coefficients_display_top_k_correct_selection(regression_train_test_split):
    """Check that top_k correctly selects features per estimator with largest
    absolute coefficients."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    top_k = min(3, full_frame["feature"].nunique())

    # Calculate expected top k features per estimator
    expected_features = set()
    for _, group in full_frame.groupby("estimator"):
        abs_coefs = group["coefficients"].abs()
        top_features = group.loc[abs_coefs.nlargest(top_k).index, "feature"].tolist()
        expected_features.update(top_features)

    display.plot(top_k=top_k, include_intercept=False)

    # Check the actual plotted data
    # For comparison reports, may have multiple subplots
    axes = (
        [display.ax_]
        if not isinstance(display.ax_, np.ndarray)
        else display.ax_.flatten()
    )

    plotted_features = set()
    for ax in axes:
        plotted_features.update(label.get_text() for label in ax.get_yticklabels())

    assert plotted_features == expected_features


def test_coefficients_display_bottom_k_correct_selection(regression_train_test_split):
    """Check that bottom_k correctly selects features per estimator with smallest
    absolute coefficients."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    bottom_k = min(3, full_frame["feature"].nunique())

    # Calculate expected bottom k features per estimator
    expected_features = set()
    for _, group in full_frame.groupby("estimator"):
        abs_coefs = group["coefficients"].abs()
        bottom_features = group.loc[
            abs_coefs.nsmallest(bottom_k).index, "feature"
        ].tolist()
        expected_features.update(bottom_features)

    display.plot(bottom_k=bottom_k, include_intercept=False)

    # Check the actual plotted data
    axes = (
        [display.ax_]
        if not isinstance(display.ax_, np.ndarray)
        else display.ax_.flatten()
    )

    plotted_features = set()
    for ax in axes:
        plotted_features.update(label.get_text() for label in ax.get_yticklabels())

    assert plotted_features == expected_features
