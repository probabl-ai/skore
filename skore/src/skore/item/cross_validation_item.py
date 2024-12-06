"""CrossValidationItem class.

This class represents the output of a cross-validation workflow.
"""

from __future__ import annotations

import contextlib
import hashlib
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy
import plotly.graph_objects
import plotly.io

from skore.item.item import Item, ItemTypeError

if TYPE_CHECKING:
    import sklearn.base

    CrossValidationReporter = Any


def _hash_numpy(arr: numpy.ndarray) -> str:
    """Compute a hash string from a numpy array.

    Parameters
    ----------
    arr : numpy array
        The numpy array whose hash will be computed.

    Returns
    -------
    hash : str
        A hash corresponding to the input array.
    """
    return hashlib.sha256(bytes(memoryview(arr))).hexdigest()


# Data used for training, passed as input to scikit-learn
Data = Any
# Target used for training, passed as input to scikit-learn
Target = Any


class CrossValidationItem(Item):
    """
    A class to represent the output of a cross-validation workflow.

    This class encapsulates the output of the
    :func:`sklearn.model_selection.cross_validate` function along with its creation and
    update timestamps.
    """

    def __init__(
        self,
        cv_results_serialized: dict,
        estimator_info: dict,
        X_info: dict,
        y_info: dict,
        plot_bytes: bytes,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        """
        Initialize a CrossValidationItem.

        Parameters
        ----------
        cv_results_serialized : dict
            The dict output of the :func:`sklearn.model_selection.cross_validate`
            function, in a form suitable for serialization.
        estimator_info : dict
            The estimator that was cross-validated.
        X_info : dict
            A summary of the data, input of the
            :func:`sklearn.model_selection.cross_validate` function.
        y_info : dict
            A summary of the target, input of the
            :func:`sklearn.model_selection.cross_validate` function.
        plot_bytes : bytes
            A plot of the cross-validation results, in the form of bytes.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        """
        super().__init__(created_at, updated_at)

        self.cv_results_serialized = cv_results_serialized
        self.estimator_info = estimator_info
        self.X_info = X_info
        self.y_info = y_info
        self.plot_bytes = plot_bytes

    @classmethod
    def factory(cls, *args, **kwargs):
        """
        Create a new CrossValidationItem instance.

        Redirects to one of the underlying `factor_*` class methods depending
        on the arguments

        Returns
        -------
        CrossValidationItem
            A new CrossValidationItem instance.
        """
        if args:
            with contextlib.suppress(ItemTypeError):
                return cls.factory_cross_validation_reporter(args[0])

        return cls.factory_raw(*args, **kwargs)

    @classmethod
    def factory_cross_validation_reporter(cls, reporter: CrossValidationReporter):
        """
        Create a new CrossValidationItem instance from a CrossValidationReporter.

        Parameters
        ----------
        reporter : CrossValidationReporter

        Returns
        -------
        CrossValidationItem
            A new CrossValidationItem instance.
        """
        if reporter.__class__.__name__ != "CrossValidationReporter":
            raise ItemTypeError(
                f"Type '{reporter.__class__}' is not supported, "
                "only 'CrossValidationReporter' is."
            )
        return cls.factory_raw(
            cv_results=reporter._cv_results,
            estimator=reporter.estimator,
            X=reporter.X,
            y=reporter.y,
            plot=reporter.plot,
        )

    @classmethod
    def factory_raw(
        cls,
        cv_results: dict,
        estimator: sklearn.base.BaseEstimator,
        X: Data,
        y: Target | None,
        plot: plotly.graph_objects.Figure,
    ) -> CrossValidationItem:
        """
        Create a new ``CrossValidationItem`` instance.

        Parameters
        ----------
        cv_results : dict
            The dict output of scikit-learn's cross_validate function.
        estimator : sklearn.base.BaseEstimator,
            The estimator that was cross-validated.
        X
            The data, input of the :func:`sklearn.model_selection.cross_validate`
            function.
        y
            The target, input of the :func:`sklearn.model_selection.cross_validate`
            function.
        plot_bytes : plotly.graph_objects.Figure
            A plot of the cross-validation results.

        Returns
        -------
        CrossValidationItem
            A new CrossValidationItem instance.
        """
        if not isinstance(cv_results, dict):
            raise ItemTypeError(f"Type '{cv_results.__class__}' is not supported.")

        cv_results_serialized = {}
        for k, v in cv_results.items():
            if k == "estimator":
                continue
            if k == "indices":
                cv_results_serialized["indices"] = {
                    "train": tuple(arr.tolist() for arr in v["train"]),
                    "test": tuple(arr.tolist() for arr in v["test"]),
                }
            if isinstance(v, numpy.ndarray):
                cv_results_serialized[k] = v.tolist()

        estimator_info = {
            "name": estimator.__class__.__name__,
            "params": repr(estimator.get_params()),
        }

        y_array = y if isinstance(y, numpy.ndarray) else numpy.array(y)
        y_info = None if y is None else {"hash": _hash_numpy(y_array)}

        X_array = X if isinstance(X, numpy.ndarray) else numpy.array(X)
        X_info = {
            "nb_rows": X_array.shape[0],
            "nb_cols": X_array.shape[1],
            "hash": _hash_numpy(X_array),
        }

        plot_bytes = plotly.io.to_json(plot, engine="json").encode("utf-8")

        return cls(
            cv_results_serialized=cv_results_serialized,
            estimator_info=estimator_info,
            X_info=X_info,
            y_info=y_info,
            plot_bytes=plot_bytes,
        )

    @cached_property
    def plot(self):
        """A plot of the cross-validation results."""
        return plotly.io.from_json(self.plot_bytes.decode("utf-8"))
