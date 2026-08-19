import pytest
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skore import evaluate
from skore.checks._utils import CheckNotApplicable
from skore.checks.model_checks import CheckSearchParamsToTune


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_suggests_missing_params(report_type, regression_data):
    """SKD015 tip is emitted when the search grid misses recommended params."""
    X, y = regression_data
    search = GridSearchCV(
        RandomForestRegressor(random_state=0),
        param_grid={"n_estimators": [10, 50]},
        cv=2,
    )
    report = evaluate(search, X, y, splitter=0.2 if report_type == "estimator" else 3)
    explanation = CheckSearchParamsToTune().check_function(report)
    assert explanation is not None
    assert "max_features" in explanation
    assert "min_samples_leaf" in explanation
    assert "max_depth" not in explanation
    assert "n_estimators" not in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_passes_when_all_recommended_covered(report_type, regression_data):
    """SKD015 passes when every recommended param is already searched."""
    X, y = regression_data
    search = GridSearchCV(
        Ridge(),
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        cv=2,
    )
    report = evaluate(search, X, y, splitter=0.2 if report_type == "estimator" else 3)
    assert CheckSearchParamsToTune().check_function(report) is None


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_pipeline_single_step(report_type, regression_data):
    """SKD015 strips pipeline prefixes correctly for a single tuned step."""
    X, y = regression_data
    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor())])
    search = GridSearchCV(pipe, param_grid={"rf__n_estimators": [10, 50]}, cv=2)
    report = evaluate(search, X, y, splitter=0.2 if report_type == "estimator" else 3)
    explanation = CheckSearchParamsToTune().check_function(report)
    assert explanation is not None
    assert "RandomForestRegressor" in explanation
    assert "max_features" in explanation
    assert "n_estimators" not in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_pipeline_multi_step(report_type, binary_classification_data):
    """SKD015 reports missing params for multiple pipeline steps."""
    X, y = binary_classification_data
    pipe = Pipeline(
        [
            ("pca", PCA()),
            ("rbf", RBFSampler(n_components=2)),
            ("clf", RandomForestClassifier(random_state=0, n_estimators=10)),
        ]
    )
    search = GridSearchCV(
        pipe,
        param_grid={
            "clf__min_samples_leaf": [10, 50],
            "rbf__n_components": [2, 3],
        },
        cv=2,
    )
    report = evaluate(search, X, y, splitter=0.2 if report_type == "estimator" else 3)
    explanation = CheckSearchParamsToTune().check_function(report)
    assert explanation is not None
    assert "StandardScaler" not in explanation
    assert "PCA" in explanation
    assert "n_components" in explanation
    assert "RBFSampler" in explanation
    assert "gamma" in explanation
    assert "RandomForestClassifier" in explanation
    assert "max_features" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_pipeline_flags_untuned_step(report_type, regression_data):
    """SKD015 flags pipeline steps whose params are not in the grid at all."""
    X, y = regression_data
    pipe = Pipeline([("pca", PCA()), ("ridge", Ridge())])
    search = GridSearchCV(pipe, param_grid={"ridge__alpha": [0.1, 1.0]}, cv=2)
    report = evaluate(search, X, y, splitter=0.2 if report_type == "estimator" else 3)
    explanation = CheckSearchParamsToTune().check_function(report)
    assert explanation is not None
    assert "PCA" in explanation
    assert "n_components" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_equivalent_params_not_suggested(report_type, regression_data):
    """Tuning max_depth should not suggest min_samples_leaf or min_samples_split."""
    X, y = regression_data
    search = GridSearchCV(
        RandomForestRegressor(random_state=0),
        param_grid={"max_depth": [3, 5, 10]},
        cv=2,
    )
    report = evaluate(search, X, y, splitter=0.2 if report_type == "estimator" else 3)
    explanation = CheckSearchParamsToTune().check_function(report)
    assert explanation is not None
    assert "min_samples_leaf" not in explanation
    assert "min_samples_split" not in explanation
    assert "max_features" in explanation


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_applicable_plain_estimator(report_type, regression_data):
    """SKD015 raises CheckNotApplicable on a plain (non-search) estimator."""
    X, y = regression_data
    report = evaluate(Ridge(), X, y, splitter=0.2 if report_type == "estimator" else 3)
    with pytest.raises(CheckNotApplicable, match="not a BaseSearchCV"):
        CheckSearchParamsToTune().check_function(report)


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_applicable_unknown_estimator(report_type, regression_data):
    """SKD015 raises CheckNotApplicable for estimators not in the table."""
    X, y = regression_data
    search = GridSearchCV(
        DummyRegressor(),
        param_grid={"strategy": ["mean", "median"]},
        cv=2,
    )
    report = evaluate(search, X, y, splitter=0.2 if report_type == "estimator" else 3)
    with pytest.raises(CheckNotApplicable, match="No parameter to recommend"):
        CheckSearchParamsToTune().check_function(report)


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_applicable_pipeline_with_no_recommendable_step(
    report_type, binary_classification_data
):
    """SKD015 raises when no pipeline step's class is in the recommendation table."""
    X, y = binary_classification_data
    pipe = Pipeline([("scaler", StandardScaler()), ("nb", GaussianNB())])
    search = GridSearchCV(pipe, param_grid={"nb__var_smoothing": [1e-9, 1e-8]}, cv=3)
    report = evaluate(search, X, y, splitter=0.2 if report_type == "estimator" else 3)
    with pytest.raises(CheckNotApplicable, match="No parameter to recommend"):
        CheckSearchParamsToTune().check_function(report)
