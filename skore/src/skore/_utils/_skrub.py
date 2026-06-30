from __future__ import annotations

import functools
from collections.abc import Iterator
from typing import Any, TypeGuard

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import NotFittedError, check_is_fitted
from skrub import DataOp, SkrubLearner, as_data_op
from skrub._data_ops._data_ops import Apply
from skrub._data_ops._evaluation import _DataOpTraversal, find_first_apply

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


def _supervised_apply_node(data_op: DataOp) -> Apply:
    """Return the supervised ``.skb.apply(estimator, y=...)`` node.

    A supervised apply is a call to :meth:`~skrub.DataOp.skb.apply` where the
    target ``y`` is passed (typically ``y=skrub.y()``).
    """
    apply_node = find_first_apply(data_op)
    if apply_node is None:
        raise NotFittedError("No supervised apply step found in the skrub learner.")
    _ensure_single_supervised_apply(data_op)
    return apply_node


def _ensure_single_supervised_apply(data_op: DataOp) -> None:
    supervised_applies: list[DataOp] = []

    def check_node(node: DataOp) -> bool:
        impl = getattr(node, "_skrub_impl", None)
        if isinstance(impl, Apply) and impl.y is not None:
            supervised_applies.append(node)
        return False

    data_op.skb.find(check_node)
    if len(supervised_applies) > 1:
        raise ValueError(
            "The skrub learner has multiple supervised apply steps; checks that "
            "inspect a single predictor are not supported."
        )


def _fitted_predictor_from_apply(apply_node: Apply) -> BaseEstimator:
    impl = apply_node._skrub_impl
    if not hasattr(impl, "estimator_"):
        raise NotFittedError(
            "The skrub learner has not been fitted. Call fit() before inspecting "
            "fitted sub-estimators."
        )
    predictor = impl.estimator_
    if isinstance(predictor, Pipeline):
        return predictor.steps[-1][1]
    return predictor


def get_predictor_and_input(
    learner: SkrubLearner, env: dict
) -> tuple[Any, BaseEstimator]:
    """Return ``(predictor_input, fitted_predictor)`` for a fitted learner.

    Uses the supervised ``.skb.apply(estimator, y=...)`` step (see
    :func:`_supervised_apply_node`). ``env`` must be a full user environment
    (for example ``report.train_data``, ``report.test_data``, or
    ``CrossValidationReport.input_data``), not a reconstructed X/y-only dict.

    The returned input is the value seen by the fitted predictor after all
    upstream graph steps have been evaluated on ``env``.
    """
    apply_node = _supervised_apply_node(learner.data_op)
    impl = apply_node._skrub_impl
    predictor_input_node = impl.X
    truncated = learner.truncated_after(
        lambda node, target=predictor_input_node: node is target
    )
    input_value = truncated.transform(env)
    return input_value, _fitted_predictor_from_apply(apply_node)


def find_fitted_estimators(learner: SkrubLearner) -> list[BaseEstimator]:
    """Return all fitted estimators from ``.skb.apply`` nodes in the learner graph."""
    fitted_estimators: dict[int, BaseEstimator] = {}

    def check_node(node: DataOp) -> bool:
        try:
            estimator = node._skrub_impl.estimator_
        except AttributeError:
            pass
        else:
            fitted_estimators[id(estimator)] = estimator
        return False

    learner.data_op.skb.find(check_node)
    return list(fitted_estimators.values())


class _FindEstimators(_DataOpTraversal):
    def __init__(self, *, include_nested: bool) -> None:
        self.include_nested = include_nested
        self.estimators: dict[int, BaseEstimator] = {}

    def handle_estimator(self, estimator: BaseEstimator):
        self.estimators[id(estimator)] = estimator
        if not self.include_nested:
            return estimator
        return (yield from super().handle_estimator(estimator))


def find_estimators(
    data_op: DataOp, *, include_nested: bool = False
) -> list[BaseEstimator]:
    """Return unfitted estimators referenced in a DataOp graph."""
    finder = _FindEstimators(include_nested=include_nested)
    finder.run(data_op)
    return list(finder.estimators.values())


def is_tunable(obj: object) -> bool:
    """Return whether obj contains skrub parameter choices rather than a fixed value."""
    return as_data_op(obj).skb.find(lambda o: not isinstance(o, DataOp)) is not None


def iter_fitted_estimator_steps(
    estimator: EstimatorLike,
) -> Iterator[tuple[str, BaseEstimator]]:
    """Yield ``(class_name, fitted_estimator)`` for hyperparameter checks.

    For :class:`~skrub.SkrubLearner`, walks the full DataOp graph and yields every
    fitted estimator from a ``.skb.apply()`` step (not only linear chains). Pipeline
    steps are expanded. Used by check SKD016.
    """
    if is_skrub_learner(estimator):
        estimators = find_fitted_estimators(estimator)
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
    ``.skb.apply(estimator, y=...)`` step, or the last step when that object is a
    :class:`~sklearn.pipeline.Pipeline` (e.g. :func:`~skrub.tabular_pipeline`).
    """
    if is_skrub_learner(estimator):
        return _fitted_predictor_from_apply(_supervised_apply_node(estimator.data_op))
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
