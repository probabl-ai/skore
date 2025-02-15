from typing import Callable, Union

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


def _check_supported_estimator(
    supported_estimators: Union[type[BaseEstimator], tuple[type[BaseEstimator], ...]],
) -> Callable:
    def check(accessor: object) -> bool:
        estimator: BaseEstimator = accessor._parent.estimator_
        if isinstance(estimator, Pipeline):
            estimator = estimator.steps[-1][1]
        supported_estimator: bool = isinstance(estimator, supported_estimators)

        if not supported_estimator:
            raise AttributeError(
                f"The {estimator.__class__.__name__} estimator is not supported "
                "by the function called."
            )

        return True

    return check
