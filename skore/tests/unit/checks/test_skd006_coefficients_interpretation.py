import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skrub import tabular_pipeline

from skore import evaluate


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_detects_unscaled_coefficients(report_type, regression_data):
    """SKD006 warns that coefficients aren't comparable when features are unscaled."""
    X, y = regression_data
    report = evaluate(
        LinearRegression(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD006" in tips.index
    assert "Features are not on the same scale" in tips.loc["SKD006", "explanation"]


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_detects_standardized_coefficients(report_type, regression_data):
    """SKD006 notes coefficients are comparable when features are standardized."""
    X, y = regression_data
    X /= X.std(axis=0)
    report = evaluate(
        LinearRegression(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD006" in tips.index
    assert "Features appear to be standardized" in tips.loc["SKD006", "explanation"]


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
@pytest.mark.filterwarnings(
    "ignore:Only pandas and polars DataFrames are supported:UserWarning:skrub"
)
def test_tabular_pipeline_with_numpy_X(report_type, regression_data):
    """SKD006 runs when tabular_pipeline is evaluated on raw numpy features."""
    X, y = regression_data
    report = evaluate(
        tabular_pipeline(LinearRegression()),
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD006" in tips.index


@pytest.mark.parametrize(
    "pipeline, expected_message",
    [
        (
            Pipeline([("model", LinearRegression())]),
            "Features are not on the same scale",
        ),
        (
            Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            "Features appear to be standardized",
        ),
    ],
)
def test_pipeline_coefficient_interpretation(
    regression_data, pipeline, expected_message
):
    """SKD006 tip reflects preprocessed feature scale in a pipeline."""
    X, y = regression_data
    report = evaluate(pipeline, X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD006" in tips.index
    assert expected_message in tips.loc["SKD006", "explanation"]
