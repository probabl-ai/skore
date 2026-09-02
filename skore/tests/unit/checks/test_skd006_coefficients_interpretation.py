import pytest
import skrub
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from skrub import SkrubLearner, tabular_pipeline

from skore import evaluate
from skore._checks.skd006_coefficients_interpretation import (
    CheckCoefficientsInterpretation,
)
from skore._checks.utils import CheckNotApplicable


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_detects_unscaled_coefficients(report_type, regression_data):
    """SKD006 warns that coefficients aren't comparable when features are unscaled."""
    X, y = regression_data
    report = evaluate(
        LinearRegression(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    explanation = CheckCoefficientsInterpretation().check_function(report)
    assert explanation is not None
    assert "Features are not on the same scale" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_detects_standardized_coefficients(report_type, regression_data):
    """SKD006 notes coefficients are comparable when features are standardized."""
    X, y = regression_data
    X /= X.std(axis=0)
    report = evaluate(
        LinearRegression(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    explanation = CheckCoefficientsInterpretation().check_function(report)
    assert explanation is not None
    assert "Features appear to be standardized" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
@pytest.mark.parametrize(
    "estimator",
    [
        tabular_pipeline(LinearRegression()),
        skrub.X().skb.apply(LinearRegression(), y=skrub.y()).skb.make_learner(),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:Only pandas and polars DataFrames are supported:UserWarning:skrub"
)
def test_tabular_pipeline_with_numpy_X(report_type, estimator, regression_data):
    """SKD006 runs when tabular_pipeline is evaluated on raw numpy features."""
    X, y = regression_data

    if isinstance(estimator, SkrubLearner):
        report = evaluate(
            estimator,
            data={"X": X, "y": y},
            splitter=0.2 if report_type == "estimator" else 3,
        )
    else:
        report = evaluate(
            estimator,
            X,
            y,
            splitter=0.2 if report_type == "estimator" else 3,
        )
    assert CheckCoefficientsInterpretation().check_function(report) is not None


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_applicable_for_non_linear_model(report_type, regression_data):
    """SKD006 needs a `coef_` attribute, absent on non-linear models."""
    X, y = regression_data
    report = evaluate(
        DecisionTreeRegressor(random_state=0),
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    with pytest.raises(CheckNotApplicable, match="does not have a `coef_` attribute."):
        CheckCoefficientsInterpretation().check_function(report)


@pytest.mark.parametrize(
    "estimator, expected_message",
    [
        (
            make_pipeline(LinearRegression()),
            "Features are not on the same scale",
        ),
        (
            make_pipeline(StandardScaler(), LinearRegression()),
            "Features appear to be standardized",
        ),
        (
            (
                skrub.X()
                .skb.apply(StandardScaler())
                .skb.apply(LinearRegression(), y=skrub.y())
                .skb.make_learner()
            ),
            "Features appear to be standardized",
        ),
    ],
)
def test_pipeline_coefficient_interpretation(
    regression_data, estimator, expected_message
):
    """SKD006 tip reflects preprocessed feature scale in a pipeline."""
    X, y = regression_data
    if isinstance(estimator, SkrubLearner):
        report = evaluate(estimator, data={"X": X, "y": y})
    else:
        report = evaluate(estimator, X, y)

    explanation = CheckCoefficientsInterpretation().check_function(report)
    assert explanation is not None
    assert expected_message in explanation
