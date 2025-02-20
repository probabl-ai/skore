from typing import Any, Callable

from sklearn.base import is_regressor
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


def _check_is_regressor_coef_task(accessor: Any) -> bool:
    """Check if the estimator is a regressor and holds a `coef_` attribute."""
    parent_estimator = accessor._parent.estimator_
    estimator = (
        parent_estimator.steps[-1][1]
        if isinstance(parent_estimator, Pipeline)
        else parent_estimator
    )
    if is_regressor(estimator) and hasattr(estimator, "coef_"):
        return True
    raise AttributeError(
        f"Estimator {accessor._parent.estimator_} is not a supported estimator by "
        "the function called."
    )
