from __future__ import annotations

import functools
from typing import TypeGuard

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import NotFittedError, check_is_fitted
from skrub import DataOp, SkrubLearner
from skrub._data_ops._data_ops import Apply
from skrub._data_ops._evaluation import find_first_apply

from skore._sklearn.types import EstimatorLike


def eval_X_y(data_op: DataOp, env: dict) -> dict:
    """
    Return a dict in which ``X`` and ``y`` have been materialized.

    The result is similar to what ``.skb.train_test_split`` outputs.
    It ensures we can retrieve the ground truth to compute metrics, and that ``X``
    and ``y`` are materialized, so that results will be consistent even if there is
    randomness in the production of ``X`` and ``y`` (e.g. they are unsorted database
    query results).
    """
    return data_op.skb.train_test_split(
        env,
        split_func=lambda X, y=None: (X, None) if y is None else (X, None, y, None),
    )["train"]


def is_skrub_learner(obj: EstimatorLike) -> TypeGuard[SkrubLearner]:
    """Detect if obj is a skrub learner (SkrubLearner, ParamSearch, OptunaSearch)."""
    return hasattr(obj, "__skrub_to_Xy_pipeline__")


def get_data_op(estimator: EstimatorLike) -> DataOp | None:
    """Return the DataOp backing a skrub learner, if any."""
    if isinstance(estimator, DataOp):
        return estimator
    if is_skrub_learner(estimator):
        return estimator.data_op
    return None


def data_op_has_explicit_cv(data_op: DataOp) -> bool:
    """Return whether ``mark_as_X`` was called with an explicit ``cv`` argument."""
    return data_op.skb.find_X_y().get("cv") is not None


def _collect_fitted_apply_steps(
    data_op: DataOp,
) -> list[tuple[str | None, BaseEstimator]]:
    """Return fitted estimators from the supervised apply down to ``X``.

    Skrub chains preprocessing and prediction as nested ``.skb.apply()`` nodes.
    The supervised apply is the outermost node (closest to the learner root);
    earlier applies are reached through each step's ``X`` input.
    """
    apply_node = find_first_apply(data_op)
    if apply_node is None:
        return []

    chain: list[tuple[str | None, BaseEstimator]] = []
    node: DataOp | None = apply_node
    while node is not None:
        impl = node._skrub_impl
        if not isinstance(impl, Apply):
            break
        if not hasattr(impl, "estimator_"):
            raise NotFittedError(
                "The skrub learner has not been fitted. Call fit() before inspecting "
                "fitted sub-estimators."
            )
        chain.append((impl.name, impl.estimator_))
        x_input = impl.X
        node = x_input if isinstance(x_input, DataOp) else None

    chain.reverse()
    return chain


def _pipeline_step_name(name: str | None, estimator: BaseEstimator, index: int) -> str:
    if name:
        return name
    base_name = type(estimator).__name__.lower()
    if isinstance(estimator, Pipeline):
        return base_name
    return f"{base_name}_{index}"


def resolve_fitted_sklearn_estimator(estimator: EstimatorLike) -> BaseEstimator:
    """Return the fitted scikit-learn estimator behind a skrub learner.

    For plain scikit-learn estimators, returns ``estimator`` unchanged.
    For :class:`~skrub.SkrubLearner`, walks the nested ``.skb.apply()`` chain
    from the supervised step down to ``X`` and returns either the single fitted
    estimator or a :class:`~sklearn.pipeline.Pipeline` when multiple applies are
    chained (e.g. ``StandardScaler`` then ``Ridge``).
    """
    if not is_skrub_learner(estimator):
        return estimator

    steps = _collect_fitted_apply_steps(estimator.data_op)
    if not steps:
        raise NotFittedError("No supervised apply step found in the skrub learner.")
    if len(steps) == 1:
        return steps[0][1]

    used_names: set[str] = set()
    pipeline_steps: list[tuple[str, BaseEstimator]] = []
    for index, (name, step_estimator) in enumerate(steps):
        step_name = _pipeline_step_name(name, step_estimator, index)
        while step_name in used_names:
            step_name = f"{step_name}_{index}"
        used_names.add(step_name)
        pipeline_steps.append((step_name, step_estimator))
    return Pipeline(pipeline_steps)


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
                kwargs["y"] = data["_skrub_y"]
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


def to_learner(estimator: EstimatorLike) -> _LearnerAdapter:
    return _LearnerAdapter(estimator)


def to_estimator(learner: EstimatorLike) -> BaseEstimator:
    assert isinstance(learner, _LearnerAdapter), (
        f"to_estimator is used to unwrap _LearnerAdapter wrappers, got: {learner!r}"
    )
    return learner.estimator
