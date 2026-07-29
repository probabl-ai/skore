import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from skore import evaluate
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.model_checks import CheckEstimatorNotTuned


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_fires_on_default_estimator(report_type, regression_data):
    """SKD016 fires when the estimator is left at sklearn defaults."""
    X, y = regression_data
    report = evaluate(
        RandomForestRegressor(),
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    explanation = CheckEstimatorNotTuned().check_function(report)
    assert explanation is not None
    assert "RandomForestRegressor" in explanation
    assert "max_features" in explanation
    assert "min_samples_leaf" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_passed_when_tuned(report_type, regression_data):
    """SKD016 passes once any recommended-or-other model param is set."""
    X, y = regression_data
    report = evaluate(
        RandomForestRegressor(max_depth=5),
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    assert CheckEstimatorNotTuned().check_function(report) is None


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_ignores_infrastructure(report_type, regression_data):
    """Setting only infrastructure params (random_state) still triggers SKD016."""
    X, y = regression_data
    report = evaluate(
        Ridge(random_state=42),
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    explanation = CheckEstimatorNotTuned().check_function(report)
    assert explanation is not None
    assert "alpha" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_ignores_budget_params(report_type, regression_data):
    """Raising max_iter alone still triggers SKD016."""
    X, y = regression_data
    report = evaluate(
        Ridge(max_iter=200), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    explanation = CheckEstimatorNotTuned().check_function(report)
    assert explanation is not None
    assert "alpha" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_pipeline_walks_steps(report_type, regression_data):
    """SKD016 reports only the pipeline steps that are still at defaults."""
    X, y = regression_data
    X, y = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])]), pd.Series(y)
    pipe = Pipeline([("pca", PCA()), ("ridge", Ridge(alpha=2.0))])
    report = evaluate(pipe, X, y, splitter=0.2 if report_type == "estimator" else 3)
    explanation = CheckEstimatorNotTuned().check_function(report)
    assert explanation is not None
    assert "PCA" in explanation
    assert "n_components" in explanation
    assert "Ridge" not in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_applicable_unknown_estimator(report_type, regression_data):
    """SKD016 raises CheckNotApplicable for estimators not in the table."""
    X, y = regression_data
    report = evaluate(
        DummyRegressor(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    with pytest.raises(CheckNotApplicable):
        CheckEstimatorNotTuned().check_function(report)


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_applicable_search(report_type, regression_data):
    """SKD016 defers to SKD015 when the estimator is a search."""
    X, y = regression_data
    search = GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1.0]}, cv=2)
    report = evaluate(search, X, y, splitter=0.2 if report_type == "estimator" else 3)
    with pytest.raises(CheckNotApplicable):
        CheckEstimatorNotTuned().check_function(report)
