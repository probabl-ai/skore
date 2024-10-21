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

from skore.item.item import Item

if TYPE_CHECKING:
    import sklearn.base


def plot_cross_validation(cv_results: dict) -> plotly.graph_objects.Figure:
    """Plot the result of a cross-validation run.

    Parameters
    ----------
    cv_results : dict
        The output of scikit-learn's cross_validate function.

    Returns
    -------
    plotly.graph_objects.Figure
        A plot of the cross-validation results
    """
    import pandas
    import plotly.graph_objects as go

    _cv_results = cv_results.copy()

    with contextlib.suppress(KeyError):
        del _cv_results["indices"]
        del _cv_results["estimator"]

    df = pandas.DataFrame(_cv_results)

    # df['fit_time'] = pd.to_timedelta(df['fit_time'], unit='s')

    dict_labels = {
        "fit_time": "fit_time (seconds)",
        "score_time": "score_time (seconds)",
    }

    fig = go.Figure()

    for col_i, col_name in enumerate(df.columns):
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df[col_name].values,
                name=dict_labels.get(col_name, col_name),
                visible=True if col_i == 0 else "legendonly",
            )
        )

    fig.update_xaxes(tickmode="linear", dtick=1, title_text="Split number")
    fig.update_yaxes(title_text="Value")
    fig.update_layout(title_text="Cross-validation results for each split")

    return fig


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
        plot = plot_cross_validation(cv_results_serialized)
        plot_bytes = plotly.io.to_json(plot, engine="json").encode("utf-8")

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
        return plotly.io.from_json(self.plot_bytes.decode("utf-8"))
