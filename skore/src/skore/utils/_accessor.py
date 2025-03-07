from typing import Any, Callable

from sklearn.pipeline import Pipeline


def _check_supported_ml_task(supported_ml_tasks: list[str]) -> Callable:
    def check(accessor: Any) -> bool:
        supported_task = any(
            task in accessor._parent._ml_task for task in supported_ml_tasks
        )

        if not supported_task:
            raise AttributeError(
                f"The {accessor._parent._ml_task} task is not a supported task by "
                f"function called. The supported tasks are {supported_ml_tasks}."
            )

        return True

    return check


def _check_has_coef() -> Callable:
    def check(accessor: Any) -> bool:
        """Check if the estimator has a `coef_` attribute."""
        parent_estimator = accessor._parent.estimator_
        estimator = (
            parent_estimator.steps[-1][1]
            if isinstance(parent_estimator, Pipeline)
            else parent_estimator
        )
        if hasattr(estimator, "coef_"):
            return True
        raise AttributeError(
            f"Estimator {parent_estimator} is not a supported estimator by "
            "the function called."
        )

    return check


def _check_has_feature_importances() -> Callable:
    def check(accessor: Any) -> bool:
        """Check if the estimator has a `feature_importances_` attribute."""
        parent_estimator = accessor._parent.estimator_
        estimator = (
            parent_estimator.steps[-1][1]
            if isinstance(parent_estimator, Pipeline)
            else parent_estimator
        )
        if hasattr(estimator, "feature_importances_"):
            return True
        raise AttributeError(
            f"Estimator {parent_estimator} is not a supported estimator by "
            "the function called."
        )

    return check
