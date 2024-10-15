"""CrossValidationItem class.

This class represents the output of a cross-validation workflow.
"""

from __future__ import annotations

import contextlib
import hashlib
from functools import cached_property
from typing import TYPE_CHECKING, Any

import altair
import numpy

from skore.item.item import Item

if TYPE_CHECKING:
    import sklearn.base


def plot_cross_validation(cv_results: dict) -> altair.Chart:
    """Plot the result of a cross-validation run.

    Parameters
    ----------
    cv_results : dict
        The output of scikit-learn's cross_validate function.

    Returns
    -------
    altair.Chart
        A plot of the cross-validation results
    """
    import altair
    import pandas

    _cv_results = cv_results.copy()

    with contextlib.suppress(KeyError):
        del _cv_results["indices"]
        del _cv_results["estimator"]

    df = (
        pandas.DataFrame(_cv_results)
        .reset_index(names="split")
        .melt(id_vars="split", var_name="metric", value_name="score")
    )

    input_dropdown = altair.binding_select(
        options=df["metric"].unique().tolist(), name="Metric: "
    )
    selection = altair.selection_point(
        fields=["metric"], bind=input_dropdown, value="test_score"
    )

    return (
        altair.Chart(df, title="Cross-validation scores per split")
        .mark_bar()
        .encode(
            altair.X("split:N").axis(
                title="Split number",
                labelAngle=0,
            ),
            altair.Y("score:Q").axis(
                title="Score",
                titleAngle=0,
                titleAlign="left",
                titleX=0,
                titleY=-5,
                labelLimit=300,
            ),
            tooltip=["metric:N", "split:N", "score:Q"],
        )
        .interactive()
        .add_params(selection)
        .transform_filter(selection)
        .properties(
            width=500,
            height=200,
            padding=15,
            autosize=altair.AutoSizeParams(type="pad", contains="padding"),
        )
    )


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

    This class encapsulates the output of scikit-learn's cross-validate function along
    with its creation and update timestamps.
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
            The dict output of scikit-learn's cross_validate function,
            in a form suitable for serialization.
        estimator_info : dict
            The estimator that was cross-validated.
        X_info : dict
            A summary of the data, input of scikit-learn's cross_validation function.
        y_info : dict
            A summary of the target, input of scikit-learn's cross_validation function.
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
        estimator : sklearn.base.BaseEstimator,
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

        # Keep plot itself as well as bytes so we can cache it
        plot = plot_cross_validation(cv_results)
        plot_bytes = plot.to_json().encode("utf-8")

        instance = cls(
            cv_results_serialized=cv_results_serialized,
            estimator_info=estimator_info,
            X_info=X_info,
            y_info=y_info,
            plot_bytes=plot_bytes,
        )

        # Cache plot
        instance.plot = plot

        return instance

    @cached_property
    def plot(self):
        """A plot of the cross-validation results."""
        return altair.Chart.from_json(self.plot_bytes.decode("utf-8"))

    @property
    def cv_results(self):
        """The cross-validation results."""
        return self.cv_results_serialized
