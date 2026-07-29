import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

from skore import EstimatorReport, evaluate
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.model_checks import CheckUselessFeatures


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_detects_useless_features(report_type):
    """Noise features are flagged when permutation importance is negligible."""
    X, y = make_regression(
        n_samples=300,
        n_features=6,
        n_informative=2,
        noise=0.1,
        shuffle=False,
        random_state=0,
    )
    report = evaluate(Ridge(), X, y, splitter=0.2 if report_type == "estimator" else 3)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD012" in tips.index
    explanation = tips.loc["SKD012", "explanation"]
    assert "permutation importance" in explanation
    assert "Feature #0" not in explanation
    assert "Feature #1" not in explanation
    assert "Feature #2" in explanation
    assert "Feature #3" in explanation
    assert "Feature #4" in explanation
    assert "Feature #5" in explanation


class _NoScoreRegressor(BaseEstimator):
    """A valid duck-typed regressor that lacks a `.score()` method."""

    def fit(self, X, y):
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X):
        return X @ self.coef_


def test_not_applicable_when_permutation_importance_fails(regression_data):
    """SKD012 raises when the estimator has no `.score()` for sklearn to use."""
    X, y = regression_data
    report = EstimatorReport(
        _NoScoreRegressor(),
        X_train=X[:60],
        y_train=y[:60],
        X_test=X[60:],
        y_test=y[60:],
    )
    with pytest.raises(CheckNotApplicable, match="Failed to compute permutation"):
        CheckUselessFeatures().check_function(report)
