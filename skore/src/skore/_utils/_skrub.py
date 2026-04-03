import functools
from typing import Any

import skrub
from sklearn.base import BaseEstimator
from sklearn.utils.validation import NotFittedError, check_is_fitted


def eval_X_y(data_op: skrub.DataOp, env: dict) -> dict:
    """
    Return a dict in which X and y have been materialized.

    The result is similar to .skb.train_test_split outputs.
    It ensures we can retrieve the ground truth to compute metrics, and that X
    and y are materialized so that results will be consistent even if there is
    randomness in the production of X and y (e.g. they are unsorted database
    query results).
    """
    return data_op.skb.train_test_split(
        env,
        split_func=lambda X, y=None: (X, None) if y is None else (X, None, y, None),
    )["train"]


def is_skrub_learner(obj: Any) -> bool:
    """Detect if obj is a skrub learner (SkrubLearner, ParamSearch, OptunaSearch)."""
    return hasattr(obj, "__skrub_to_Xy_pipeline__")


class _LearnerAdapter(BaseEstimator):
    """Wrap an estimator to have the learner interface (accept dicts)."""

    def __init__(self, estimator):
        self.estimator = estimator

    def __getattr__(self, name):
        if name not in [
            "fit",
            "predict",
            "decision_function",
            "predict_proba",
            "score",
        ]:
            return getattr(self.estimator, name)
        estimator_method = getattr(self.estimator, name)

        @functools.wraps(estimator_method)
        def learner_method(data):
            kwargs = {"X": data["_skrub_X"]}
            if name in ["fit", "score"]:
                kwargs["y"] = data.get("_skrub_y")
            return estimator_method(**kwargs)

        return learner_method

    def __sklearn_is_fitted__(self):
        try:
            check_is_fitted(self.estimator)
            return True
        except NotFittedError:
            return False

    def __sklearn_tags__(self):
        return self.estimator.__sklearn_tags__()


def to_learner(estimator: BaseEstimator):
    return _LearnerAdapter(estimator)


def to_estimator(learner: skrub.SkrubLearner):
    if isinstance(learner, _LearnerAdapter):
        return learner.estimator
    return learner.find_fitted_estimator("estimator")
