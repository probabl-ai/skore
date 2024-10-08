"""CrossValidationItem class.

This class represents the output of a cross-validation workflow.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from skore.item.item import Item

if TYPE_CHECKING:
    import sklearn.base


def _hash_numpy(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


# Data used for training, passed as input to scikit-learn
Data = Any
# Target used for training, passed as input to scikit-learn
Target = Any


class CrossValidationItem(Item):
    """
    A class to represent the output of a cross-validation workflow.

    This class encapsulates the output of scikit-learn's cross-validate function along
    with its creation and update timestamps.
    """

    def __init__(
        self,
        cv_results: dict,
        estimator_info: dict,
        X_info: dict,
        y_info: dict,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        """
        Initialize a CrossValidationItem.

        Parameters
        ----------
        cv_results : dict
            The dict output of scikit-learn's cross_validate function.
        estimator_info : dict
            The estimator that was cross-validated.
        X_info : dict
            Summary of the data, input of scikit-learn's cross_validation function.
        y_info : dict
            Summary of the target, input of scikit-learn's cross_validation function.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        """
        super().__init__(created_at, updated_at)

        self.cv_results = cv_results
        self.estimator_info = estimator_info
        self.X_info = X_info
        self.y_info = y_info

    @classmethod
    def factory(
        cls,
        cv_results: dict,
        estimator: sklearn.base.BaseEstimator,
        X: Data,
        y: Target | None,
    ) -> CrossValidationItem:
        """
        Create a new CrossValidationItem instance.

        Parameters
        ----------
        cv_results : dict
            The dict output of scikit-learn's cross_validate function.
        estimator: sklearn.base.BaseEstimator,
            The estimator that was cross-validated.
        X
            The data, input of scikit-learn's cross_validation function.
        y
            The target, input of scikit-learn's cross_validation function.

        Returns
        -------
        CrossValidationItem
            A new CrossValidationItem instance.
        """
        if not isinstance(cv_results, dict):
            raise TypeError(f"Type '{cv_results.__class__}' is not supported.")

        estimator_info = {
            "name": estimator.__class__.__name__,
            "params": estimator.get_params(),
        }

        y_info = None if y is None else {"hash": _hash_numpy(y)}

        X_info = {
            "nb_rows": X.shape[0],
            "nb_cols": X.shape[1],
            "hash": _hash_numpy(X),
        }

        instance = cls(
            cv_results=cv_results,
            estimator_info=estimator_info,
            X_info=X_info,
            y_info=y_info,
        )

        return instance
