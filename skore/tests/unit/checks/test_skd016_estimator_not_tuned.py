import pandas as pd
import pytest
import skrub
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from skrub import SkrubLearner, tabular_pipeline

from skore import evaluate
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.model_checks import CheckEstimatorNotTuned


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
@pytest.mark.parametrize(
    "estimator",
    [
        RandomForestRegressor(random_state=0),
        tabular_pipeline(RandomForestRegressor(random_state=0)),
        (
            skrub.X()
            .skb.apply(RandomForestRegressor(random_state=0), y=skrub.y())
            .skb.make_learner()
        ),
    ],
)
def test_fires_on_default_estimator(report_type, estimator, regression_data):
    """SKD016 fires when the estimator is left at sklearn defaults."""
    X, y = regression_data
    if isinstance(estimator, SkrubLearner):
        report = evaluate(
            estimator,
            data={"X": X, "y": y},
            splitter=0.2 if report_type == "estimator" else 3,
        )
    else:
        report = evaluate(
            estimator, X, y, splitter=0.2 if report_type == "estimator" else 3
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


def test_skrub_table_vectorizer_chain(regression_data):
    """SKD016 walks fitted estimators from a non-linear skrub apply graph."""
    X, y = regression_data
    learner = (
        skrub.X()
        .skb.apply(skrub.TableVectorizer())
        .skb.apply(Ridge(), y=skrub.y())
        .skb.make_learner()
    )
    report = evaluate(learner, data={"X": X, "y": y})
    explanation = CheckEstimatorNotTuned().check_function(report)
    assert explanation is not None
    assert "alpha" in explanation


def test_passes_when_skrub_param_is_tunable(regression_data):
    """SKD016 passes when a recommended param is a skrub choice in the DataOp."""
    X, y = regression_data
    learner = (
        skrub.X()
        .skb.apply(Ridge(alpha=skrub.choose_from([0.1, 1.0])), y=skrub.y())
        .skb.make_learner()
    )
    report = evaluate(learner, data={"X": X, "y": y})
    assert CheckEstimatorNotTuned().check_function(report) is None


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_applicable_unknown_estimator(report_type, regression_data):
    """SKD016 raises CheckNotApplicable for estimators not in the table."""
    X, y = regression_data
    report = evaluate(
        DummyRegressor(), X, y, splitter=0.2 if report_type == "estimator" else 3
    )
    with pytest.raises(CheckNotApplicable, match="No parameter to recommend"):
        CheckEstimatorNotTuned().check_function(report)


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_applicable_search(report_type, regression_data):
    """SKD016 defers to SKD015 when the estimator is a search."""
    X, y = regression_data
    search = GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1.0]}, cv=2)
    report = evaluate(search, X, y, splitter=0.2 if report_type == "estimator" else 3)
    with pytest.raises(CheckNotApplicable, match="is a BaseSearchCV"):
        CheckEstimatorNotTuned().check_function(report)
