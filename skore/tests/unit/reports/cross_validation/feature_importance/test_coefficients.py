import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from skore import CoefficientsDisplay, CrossValidationReport


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
def test_with_model_exposing_coef(regression_data, estimator):
    """Check that we can create a coefficients display from model exposing a `coef_`
    attribute."""
    X, y = regression_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    assert hasattr(report.feature_importance, "coefficients")
    display = report.feature_importance.coefficients()
    assert isinstance(display, CoefficientsDisplay)


def test_with_model_not_exposing_coef(regression_data):
    """Check that we cannot create a coefficients display from model not exposing a
    `coef_` attribute."""
    X, y = regression_data
    report = CrossValidationReport(DecisionTreeRegressor(), X, y, splitter=2)
    assert not hasattr(report.feature_importance, "coefficients")


def test_coefficients_display_top_k_correct_selection(regression_data):
    """Check that top_k correctly selects features with largest mean absolute
    coefficients across folds."""
    X, y = regression_data
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    top_k = min(3, full_frame["feature"].nunique())

    # Calculate expected top k features based on
    # mean absolute coefficients across splits
    mean_abs_coefs = full_frame.groupby("feature")["coefficients"].apply(
        lambda x: x.abs().mean()
    )
    expected_features = set(mean_abs_coefs.nlargest(top_k).index.tolist())

    display.plot(top_k=top_k, include_intercept=False)

    # Check the actual plotted data
    ax = (
        display.ax_
        if not isinstance(display.ax_, np.ndarray)
        else display.ax_.flatten()[0]
    )
    plotted_features = set(ax.get_yticklabels())
    plotted_feature_texts = {label.get_text() for label in plotted_features}

    assert plotted_feature_texts == expected_features


def test_coefficients_display_bottom_k_correct_selection(regression_data):
    """Check that bottom_k correctly selects features with smallest mean absolute
    coefficients across folds."""
    X, y = regression_data
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    bottom_k = min(3, full_frame["feature"].nunique())

    # Calculate expected bottom k features based on
    # mean absolute coefficients across splits
    mean_abs_coefs = full_frame.groupby("feature")["coefficients"].apply(
        lambda x: x.abs().mean()
    )
    expected_features = set(mean_abs_coefs.nsmallest(bottom_k).index.tolist())

    display.plot(bottom_k=bottom_k, include_intercept=False)

    # Check the actual plotted data
    ax = (
        display.ax_
        if not isinstance(display.ax_, np.ndarray)
        else display.ax_.flatten()[0]
    )
    plotted_features = set(ax.get_yticklabels())
    plotted_feature_texts = {label.get_text() for label in plotted_features}

    assert plotted_feature_texts == expected_features
