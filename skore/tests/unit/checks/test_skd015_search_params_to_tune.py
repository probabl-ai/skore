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
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.model_checks import CheckSearchParamsToTune


def test_suggests_missing_params(regression_data):
    """SKD015 tip is emitted when the search grid misses recommended params."""
    X, y = regression_data
    search = GridSearchCV(
        RandomForestRegressor(random_state=0),
        param_grid={"n_estimators": [10, 50]},
        cv=2,
    )
    report = evaluate(search, X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD015" in tips.index
    explanation = tips.loc["SKD015", "explanation"]
    assert "max_features" in explanation
    assert "min_samples_leaf" in explanation
    assert "max_depth" not in explanation
    assert "n_estimators" not in explanation


def test_suggests_missing_params_on_cv_report(regression_data):
    """SKD015 tip is emitted on a cross-validation report."""
    X, y = regression_data
    search = GridSearchCV(
        RandomForestRegressor(random_state=0),
        param_grid={"n_estimators": [10, 50]},
        cv=2,
    )
    report = evaluate(search, X, y, splitter=3)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD015" in tips.index
    assert "max_features" in tips.loc["SKD015", "explanation"]


def test_passes_when_all_recommended_covered(regression_data):
    """SKD015 passes when every recommended param is already searched."""
    X, y = regression_data
    search = GridSearchCV(
        Ridge(),
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        cv=2,
    )
    report = evaluate(search, X, y)
    summary = report.checks.summarize()
    assert "SKD015" in set(summary.frame(section="passed")["code"])


def test_pipeline_single_step(regression_data):
    """SKD015 strips pipeline prefixes correctly for a single tuned step."""
    X, y = regression_data
    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor())])
    search = GridSearchCV(pipe, param_grid={"rf__n_estimators": [10, 50]}, cv=2)
    report = evaluate(search, X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD015" in tips.index
    explanation = tips.loc["SKD015", "explanation"]
    assert "RandomForestRegressor" in explanation
    assert "max_features" in explanation
    assert "n_estimators" not in explanation


def test_pipeline_multi_step(binary_classification_data):
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
    report = evaluate(search, X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD015" in tips.index
    explanation = tips.loc["SKD015", "explanation"]
    assert "StandardScaler" not in explanation
    assert "PCA" in explanation
    assert "n_components" in explanation
    assert "RBFSampler" in explanation
    assert "gamma" in explanation
    assert "RandomForestClassifier" in explanation
    assert "max_features" in explanation


def test_pipeline_flags_untuned_step(regression_data):
    """SKD015 flags pipeline steps whose params are not in the grid at all."""
    X, y = regression_data
    pipe = Pipeline([("pca", PCA()), ("ridge", Ridge())])
    search = GridSearchCV(pipe, param_grid={"ridge__alpha": [0.1, 1.0]}, cv=2)
    report = evaluate(search, X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD015" in tips.index
    explanation = tips.loc["SKD015", "explanation"]
    assert "PCA" in explanation
    assert "n_components" in explanation


def test_equivalent_params_not_suggested(regression_data):
    """Tuning max_depth should not suggest min_samples_leaf or min_samples_split."""
    X, y = regression_data
    search = GridSearchCV(
        RandomForestRegressor(random_state=0),
        param_grid={"max_depth": [3, 5, 10]},
        cv=2,
    )
    report = evaluate(search, X, y)
    tips = report.checks.summarize().frame(section="tip").set_index("code")
    assert "SKD015" in tips.index
    explanation = tips.loc["SKD015", "explanation"]
    assert "min_samples_leaf" not in explanation
    assert "min_samples_split" not in explanation
    assert "max_features" in explanation


def test_not_applicable_plain_estimator(regression_data):
    """SKD015 raises CheckNotApplicable on a plain (non-search) estimator."""
    X, y = regression_data
    report = evaluate(Ridge(), X, y)
    with pytest.raises(CheckNotApplicable):
        CheckSearchParamsToTune().check_function(report)


def test_not_applicable_plain_estimator_on_cv_report(regression_data):
    """SKD015 raises CheckNotApplicable with a plain estimator on a CV report."""
    X, y = regression_data
    report = evaluate(Ridge(), X, y, splitter=3)
    with pytest.raises(CheckNotApplicable, match="not a BaseSearchCV"):
        CheckSearchParamsToTune().check_function(report)


def test_not_applicable_unknown_estimator(regression_data):
    """SKD015 raises CheckNotApplicable for estimators not in the table."""
    X, y = regression_data
    search = GridSearchCV(
        DummyRegressor(),
        param_grid={"strategy": ["mean", "median"]},
        cv=2,
    )
    report = evaluate(search, X, y)
    with pytest.raises(CheckNotApplicable):
        CheckSearchParamsToTune().check_function(report)


def test_not_applicable_pipeline_with_no_recommendable_step(
    binary_classification_data,
):
    """SKD015 raises when no pipeline step's class is in the recommendation table."""
    X, y = binary_classification_data
    pipe = Pipeline([("scaler", StandardScaler()), ("nb", GaussianNB())])
    search = GridSearchCV(pipe, param_grid={"nb__var_smoothing": [1e-9, 1e-8]}, cv=3)
    report = evaluate(search, X, y)
    with pytest.raises(CheckNotApplicable, match="No parameter to recommend"):
        CheckSearchParamsToTune().check_function(report)
