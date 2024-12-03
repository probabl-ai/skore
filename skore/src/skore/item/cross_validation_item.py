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


def plot_cross_validation_aggregation(
    cv_results_items_versions: list[CrossValidationItem],
) -> plotly.graph_objects.Figure:
    """Plot the result of the aggregation of several cross-validation runs.

    Parameters
    ----------
    cv_results_items_versions : list[CrossValidationItem]
        A list of outputs of scikit-learn's cross_validate function.

    Returns
    -------
    plotly.graph_objects.Figure
        A plot of the aggregated cross-validation results
    """
    from datetime import timedelta

    import pandas
    import plotly.graph_objects as go

    _cv_results = cv_results_items_versions.copy()

    df = pandas.DataFrame([v.cv_results_serialized for v in _cv_results])
    df = df.drop(columns=["indices", "estimator"], errors="ignore")
    df = df.apply(pandas.Series.explode)
    df = df.reset_index(names="run_number")

    # Move time columns to last and "test_score" to first
    if "fit_time" in df.columns:
        df.insert(len(df.columns) - 1, "fit_time", df.pop("fit_time"))
    if "score_time" in df.columns:
        df.insert(len(df.columns) - 1, "score_time", df.pop("score_time"))
    if "test_score" in df.columns:
        df.insert(1, "test_score", df.pop("test_score"))

    dict_labels = {
        "fit_time": "fit_time (seconds)",
        "score_time": "score_time (seconds)",
    }

    fig = go.Figure()

    for col_i, col_name in enumerate(df.columns[1:]):
        metric_name = dict_labels.get(col_name, col_name)
        bar_color = plotly.colors.qualitative.Plotly[
            col_i % len(plotly.colors.qualitative.Plotly)
        ]

        common_kwargs = dict(
            visible=True if col_i == 1 else "legendonly",
            legendgroup=f"group{col_i}",
            # If the metric is a duration (e.g. "fit_time"),
            # we show a different hover text
            hovertemplate=(
                "%{customdata}" f"<extra>{col_name} (timedelta)</extra>"
                if col_name.endswith("_time")
                else "%{y}"
            ),
            customdata=(
                [str(timedelta(seconds=x)) for x in df[col_name].values]
                if col_name.endswith("_time")
                else None
            ),
        )

        fig.add_traces(
            [
                go.Scatter(
                    x=df["run_number"],
                    y=df[col_name].values,
                    name=metric_name,
                    mode="markers",
                    marker_color=bar_color,
                    showlegend=True,
                    **common_kwargs,
                ),
            ]
        )

    fig.update_xaxes(tickmode="linear", dtick=1, title_text="Run number")
    fig.update_yaxes(title_text="Value")
    fig.update_layout(title_text="Cross-validation results for each run")

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
                "Arguments to CrossValidationItem.factory are not supported."
            )
        return CrossValidationItem.factory_raw(
            cv_results=reporter._cv_results,
            estimator=reporter.estimator,
            X=reporter.X,
            y=reporter.y,
            plot_bytes=plotly.io.to_json(reporter.plot, engine="json").encode("utf-8"),
        )

    @classmethod
    def factory_raw(
        cls,
        cv_results: dict,
        estimator: sklearn.base.BaseEstimator,
        X: Data,
        y: Target | None,
        plot_bytes: bytes,
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
        plot_bytes : bytes
            A plot of the cross-validation results, in bytes form.
            The plot is assumed to have been made with plotly.

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


class CrossValidationAggregationItem(Item):
    """Aggregated outputs of several cross-validation workflow runs."""

    def __init__(
        self,
        plot_bytes: bytes,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        """
        Initialize an CrossValidationAggregationItem.

        Parameters
        ----------
        plot_bytes : bytes
            A plot of the aggregated cross-validation results, in the form of bytes.
        created_at : str
            The creation timestamp in ISO format.
        updated_at : str
            The last update timestamp in ISO format.
        """
        super().__init__(created_at, updated_at)

        self.plot_bytes = plot_bytes

    @classmethod
    def factory(
        cls,
        cv_results_items_versions: list[CrossValidationItem],
    ) -> CrossValidationAggregationItem:
        """
        Create a new CrossValidationAggregationItem instance.

        Parameters
        ----------
        cv_results_items_versions: list[CrossValidationItem]
            A list of cross_validate items to be aggregated.

        Returns
        -------
        CrossValidationAggregationItem
            A new CrossValidationAggregationItem instance.
        """
        plot = plot_cross_validation_aggregation(cv_results_items_versions)

        plot_bytes = plotly.io.to_json(plot, engine="json").encode("utf-8")

        instance = cls(
            plot_bytes=plot_bytes,
        )

        # Cache plot
        instance.plot = plot

        return instance

    @cached_property
    def plot(self):
        """An aggregation plot of all the cross-validation results.

        Results are shown from the oldest to the current.
        """
        return plotly.io.from_json(self.plot_bytes.decode("utf-8"))
