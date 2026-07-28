import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skrub import tabular_pipeline

from skore import evaluate


def test_detects_coefficient_interpretation(regression_data):
    """Check that the coefficient interpretation tip is emitted."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD006" in tips.index
    assert "Features are not on the same scale" in tips.loc["SKD006", "explanation"]

    X /= X.std(axis=0)
    report = evaluate(LinearRegression(), X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD006" in tips.index
    assert "Features appear to be standardized" in tips.loc["SKD006", "explanation"]


def test_detects_coefficient_interpretation_cv(regression_data):
    """Check that the coefficient interpretation tip is emitted on a CV report."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=3)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD006" in tips.index
    assert "Features are not on the same scale" in tips.loc["SKD006", "explanation"]

    X /= X.std(axis=0)
    report = evaluate(LinearRegression(), X, y, splitter=3)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD006" in tips.index
    assert "Features appear to be standardized" in tips.loc["SKD006", "explanation"]


@pytest.mark.filterwarnings(
    "ignore:Only pandas and polars DataFrames are supported:UserWarning:skrub"
)
def test_tabular_pipeline_with_numpy_X(regression_data):
    """SKD006 runs when tabular_pipeline is evaluated on raw numpy features."""
    X, y = regression_data
    report = evaluate(tabular_pipeline(LinearRegression()), X, y, splitter=0.2)
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
