from __future__ import annotations

import functools
from collections.abc import Iterator
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


def _supervised_apply_node(data_op: DataOp) -> DataOp:
    apply_node = find_first_apply(data_op)
    if apply_node is None:
        raise NotFittedError("No supervised apply step found in the skrub learner.")
    return apply_node


def _supervised_fitted_estimator(learner: SkrubLearner) -> BaseEstimator:
    impl = _supervised_apply_node(learner.data_op)._skrub_impl
    if not isinstance(impl, Apply):
        raise TypeError(
            f"The supervised step does not represent an estimator application: "
            f"{learner.data_op!r}"
        )
    if not hasattr(impl, "estimator_"):
        raise NotFittedError(
            "The skrub learner has not been fitted. Call fit() before inspecting "
            "fitted sub-estimators."
        )
    return impl.estimator_


def get_preprocess_apply_node(data_op: DataOp) -> DataOp | None:
    """Return the preprocessing .skb.apply() node before the supervised step, if any."""
    supervised_impl = _supervised_apply_node(data_op)._skrub_impl
    if not isinstance(supervised_impl, Apply):
        return None
    x_input = supervised_impl.X
    if isinstance(x_input, DataOp) and isinstance(x_input._skrub_impl, Apply):
        return x_input
    return None


def collect_fitted_apply_estimators(data_op: DataOp) -> list[BaseEstimator]:
    """Return fitted estimators along nested applies from preprocessing to predictor."""
    apply_node = _supervised_apply_node(data_op)
    chain: list[BaseEstimator] = []
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
        chain.append(impl.estimator_)
        x_input = impl.X
        node = x_input if isinstance(x_input, DataOp) else None
    chain.reverse()
    return chain


def iter_fitted_estimator_steps(
    estimator: EstimatorLike,
) -> Iterator[tuple[str, BaseEstimator]]:
    """Yield ``(class_name, fitted_estimator)`` from a learner or plain estimator."""
    if is_skrub_learner(estimator):
        estimators = collect_fitted_apply_estimators(estimator.data_op)
    else:
        estimators = [estimator]

    for fitted in estimators:
        if isinstance(fitted, Pipeline):
            for _, step in fitted.steps:
                yield type(step).__name__, step
        else:
            yield type(fitted).__name__, fitted


def resolve_fitted_predictor(estimator: EstimatorLike) -> BaseEstimator:
    """Return the fitted predictor behind ``estimator``.

    For :class:`~skrub.SkrubLearner`, returns the estimator fitted in the supervised
    ``.skb.apply()`` step, or the last step when that object is a
    :class:`~sklearn.pipeline.Pipeline` (e.g. :func:`~skrub.tabular_pipeline`).
    """
    if is_skrub_learner(estimator):
        fitted = _supervised_fitted_estimator(estimator)
        if isinstance(fitted, Pipeline):
            return fitted.steps[-1][1]
        return fitted

    if isinstance(estimator, Pipeline):
        return estimator.steps[-1][1]
    return estimator


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
