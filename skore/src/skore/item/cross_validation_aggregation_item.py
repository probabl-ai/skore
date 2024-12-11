"""CrossValidationAggregationItem class.

This class represents the aggregation of several cross-validation runs.
"""

from __future__ import annotations

from functools import cached_property

import plotly.graph_objects
import plotly.io

from skore.item.cross_validation_item import CrossValidationItem
from skore.item.item import Item


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
