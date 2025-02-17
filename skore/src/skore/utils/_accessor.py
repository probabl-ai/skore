from typing import Any, Callable


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


def _check_is_linear_regression(accessor):
    """Check if the estimator is a linear regression, Ridge, or Lasso."""
    from sklearn.linear_model import Lasso, LinearRegression, Ridge

    if isinstance(accessor._parent.estimator_, (LinearRegression, Ridge, Lasso)):
        return True
    raise AttributeError(
        f"Estimator {accessor._parent.estimator_} is not a supported estimator by "
        "the function called."
    )
