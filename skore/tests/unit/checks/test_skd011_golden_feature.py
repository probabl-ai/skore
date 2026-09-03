import numpy as np
import pandas as pd
import pytest
import skrub
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skrub import SkrubLearner, tabular_pipeline

from skore import evaluate
from skore._checks.skd011_golden_feature import CheckGoldenFeature
from skore._checks.utils import CheckNotApplicable


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        tabular_pipeline(LinearRegression()),
        skrub.X().skb.apply(LinearRegression(), y=skrub.y()).skb.make_learner(),
    ],
)
def test_detects_golden_feature(report_type, estimator):
    """Features correlated with the target get flagged as golden."""
    rng = np.random.RandomState(0)
    n = 200
    y = rng.normal(size=n)
    # Features 2 and 3 are unrelated to y
    X = pd.DataFrame(
        {
            "Feature 0": y * 10,
            "Feature 1": y + rng.normal(scale=0.01, size=n),
            "Feature 2": rng.normal(size=n),
            "Feature 3": rng.normal(size=n),
        }
    )
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
    explanation = CheckGoldenFeature().check_function(report)

    assert explanation is not None
    assert "Feature 0" in explanation
    assert "Feature 1" in explanation
    assert "Feature 2" not in explanation
    assert "Feature 3" not in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_sklearn_pipeline_preserves_feature_names(report_type):
    """SKD011 works when a sklearn preprocessor returns an ndarray.

    Default sklearn ``transform`` drops column names; ``get_preprocessed_X``
    must restore them so single-feature selection matches ``_get_feature_names``.
    """
    rng = np.random.RandomState(0)
    n = 200
    y = rng.normal(size=n)
    # Features 2 and 3 are unrelated to y
    X = pd.DataFrame(
        {
            "col_0": y * 10,
            "col_1": y + rng.normal(scale=0.01, size=n),
            "col_2": rng.normal(size=n),
            "col_3": rng.normal(size=n),
        }
    )
    report = evaluate(
        make_pipeline(StandardScaler(), LinearRegression()),
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    explanation = CheckGoldenFeature().check_function(report)

    assert explanation is not None
    assert "col_0" in explanation
    assert "col_1" in explanation
    assert "col_2" not in explanation
    assert "col_3" not in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_applicable_single_feature_estimator(report_type, regression_data):
    """SKD011 needs at least 2 features to compare against a single one."""
    X, y = regression_data
    report = evaluate(
        LinearRegression(),
        X[:, :1],
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )

    with pytest.raises(CheckNotApplicable, match="only one feature"):
        CheckGoldenFeature().check_function(report)


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_applicable_when_single_feature_refit_fails(report_type, regression_data):
    """SKD011 raises when the single-feature estimator cannot be refit.

    ``PLSRegression(n_components=2)`` fails to fit when selected down to a
    single column, since 2 components cannot be extracted from 1 feature.
    """
    X, y = regression_data
    report = evaluate(
        PLSRegression(n_components=2),
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    with pytest.raises(
        CheckNotApplicable, match="Failed to create report from single feature."
    ):
        CheckGoldenFeature().check_function(report)
