
import pandas as pd
import pytest
import skrub
from numpy.testing import assert_array_equal
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skrub import tabular_pipeline

from skore._utils._skrub import (
    find_estimators,
    find_fitted_estimators,
    get_predictor_and_input,
    is_tunable,
    iter_fitted_estimator_steps,
    resolve_fitted_predictor,
)


@pytest.fixture
def regression_xy():
    """Small regression set as pandas objects (skrub expects DataFrames)."""
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "b": [2.0, 3, 4, 5, 6]})
    y = pd.Series([0.0, 1, 0, 1, 0])
    return df, y


def case_chained_applies():
    """A pipeline with several Apply nodes."""
    return (
        skrub.X()
        .skb.apply(StandardScaler())
        .skb.apply(Ridge(), y=skrub.y())
        .skb.make_learner()
    )


def case_tabular_pipeline():
    """A ``tabular_pipeline``."""
    return (
        skrub.X().skb.apply(tabular_pipeline(Ridge()), y=skrub.y()).skb.make_learner()
    )


def case_supervised_preprocessing():
    """A pipeline with several supervised Apply nodes."""
    skb_y = skrub.y()
    return (
        skrub.X()
        .skb.apply(SelectKBest(k=1), y=skb_y)
        .skb.apply(Ridge(), y=skb_y)
        .skb.make_learner()
    )


def case_supervised_preprocessing_choice():
    """A pipeline with supervised preprocessing Apply nodes, some of which are
    ``Choice``s (these nodes do not expose ``transform``)."""
    skb_y = skrub.y()
    return (
        skrub.X()
        .skb.apply(
            skrub.choose_from([SelectKBest(k=1), SelectKBest(k=2)], name="fs"), y=skb_y
        )
        .skb.apply(Ridge(), y=skb_y)
        .skb.make_learner()
    )


@pytest.mark.parametrize(
    "case",
    [
        case_chained_applies,
        case_tabular_pipeline,
        case_supervised_preprocessing,
        case_supervised_preprocessing_choice,
    ],
)
def test_resolve_fitted_predictor(case, regression_xy):
    """``resolve_fitted_predictor`` finds the final Ridge estimator."""
    df, y = regression_xy
    learner = case()
    learner.fit({"X": df, "y": y})

    predictor = resolve_fitted_predictor(learner)
    assert isinstance(predictor, Ridge)


def test_get_predictor_and_input_matches_sklearn_pipeline_preprocessing(regression_xy):
    """Skrub-native transform matches sklearn Pipeline preprocessing."""
    df, y = regression_xy
    learner = (
        skrub.X()
        .skb.apply(StandardScaler())
        .skb.apply(Ridge(), y=skrub.y())
        .skb.make_learner()
    )
    env = {"X": df, "y": y}
    learner.fit(env)

    skrub_X, _ = get_predictor_and_input(learner, env)

    sklearn_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    sklearn_pipe.fit(df, y)
    sklearn_X = sklearn_pipe[:-1].transform(df)

    assert_array_equal(skrub_X, sklearn_X)


def test_get_predictor_and_input_multiple_supervised_applies_raises(regression_xy):
    """Learners with several supervised applies are rejected by predictor helpers."""
    df, y = regression_xy
    learner = (
        skrub.X()
        .skb.apply(Ridge(), y=skrub.y())
        .skb.apply(Ridge(), y=skrub.var("y2", y))
        .skb.make_learner()
    )

    with pytest.raises(ValueError, match="multiple supervised apply"):
        get_predictor_and_input(learner, {"X": df, "y": y, "y2": y})


def test_find_fitted_estimators_on_table_vectorizer_chain(regression_xy):
    """Discover estimators when applies do not form a single sklearn Pipeline chain.

    ``TableVectorizer`` followed by a supervised ``Ridge`` apply is Jerome's
    canonical non-linear graph: preprocessing is not nested inside the supervised
    ``Pipeline``, so walking a single apply chain is not enough. Graph traversal
    must still find every fitted ``.skb.apply()`` step.
    """
    df, y = regression_xy
    learner = (
        skrub.X()
        .skb.apply(skrub.TableVectorizer())
        .skb.apply(Ridge(), y=skrub.y())
        .skb.make_learner()
        .fit({"X": df, "y": y})
    )

    fitted_names = {type(est).__name__ for est in find_fitted_estimators(learner)}
    step_names = {name for name, _ in iter_fitted_estimator_steps(learner)}

    assert fitted_names == step_names == {"Ridge", "ApplyToCols"}


def test_find_estimators_include_nested(regression_xy):
    """Unfitted estimator discovery can include or exclude nested sub-estimators."""
    df, y = regression_xy
    data_op = (
        skrub.X().skb.apply(skrub.TableVectorizer()).skb.apply(Ridge(), y=skrub.y())
    )

    top_level = {
        type(est).__name__ for est in find_estimators(data_op, include_nested=False)
    }
    nested = {
        type(est).__name__ for est in find_estimators(data_op, include_nested=True)
    }

    assert top_level == {"TableVectorizer", "Ridge"}
    assert "OneHotEncoder" in nested
    assert len(nested) > len(top_level)


def test_is_tunable_detects_skrub_choices():
    """Skrub parameter choices are detected as tunable."""
    assert is_tunable(skrub.choose_from([0.1, 1.0]))
    assert not is_tunable(1.0)
