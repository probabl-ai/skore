import warnings

import pandas as pd
import pytest
import skrub
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skrub import tabular_pipeline

from skore import evaluate
from skore._sklearn._checks._utils import get_preprocessed_X
from skore._utils._skrub import get_preprocess_apply_node, resolve_fitted_predictor


@pytest.fixture
def regression_xy():
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "b": [2.0, 3, 4, 5, 6]})
    y = pd.Series([0.0, 1, 0, 1, 0])
    return df, y


def test_resolve_fitted_predictor_returns_ridge_for_chained_applies(regression_xy):
    """Chained applies resolve to the supervised predictor, not a stitched Pipeline."""
    df, y = regression_xy
    learner = (
        skrub.X()
        .skb.apply(StandardScaler())
        .skb.apply(Ridge(), y=skrub.y())
        .skb.make_learner()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = evaluate(learner, data={"X": df, "y": y})

    predictor = resolve_fitted_predictor(report.estimator_)
    assert isinstance(predictor, Ridge)


def test_resolve_fitted_predictor_returns_inner_pipeline_last_step_for_tabular(
    regression_xy,
):
    """A single apply wrapping tabular_pipeline resolves to the inner predictor."""
    df, y = regression_xy
    learner = (
        skrub.X().skb.apply(tabular_pipeline(Ridge()), y=skrub.y()).skb.make_learner()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = evaluate(learner, data={"X": df, "y": y})

    predictor = resolve_fitted_predictor(report.estimator_)
    assert isinstance(predictor, Ridge)


def test_get_preprocessed_X_matches_sklearn_pipeline_preprocessing(regression_xy):
    """Skrub-native transform matches sklearn Pipeline preprocessing."""
    df, y = regression_xy
    learner = (
        skrub.X()
        .skb.apply(StandardScaler())
        .skb.apply(Ridge(), y=skrub.y())
        .skb.make_learner()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = evaluate(learner, data={"X": df, "y": y})

    assert get_preprocess_apply_node(report.estimator_.data_op) is not None
    skrub_X = get_preprocessed_X(report, data_source="train")
    train_X = report.train_data["_skrub_X"]
    train_y = report.train_data["_skrub_y"]
    sklearn_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    sklearn_pipe.fit(train_X, train_y)
    sklearn_X = sklearn_pipe[:-1].transform(train_X)
    pd.testing.assert_frame_equal(
        pd.DataFrame(skrub_X, columns=train_X.columns).reset_index(drop=True),
        pd.DataFrame(sklearn_X, columns=train_X.columns).reset_index(drop=True),
        rtol=1e-5,
        atol=1e-5,
    )
