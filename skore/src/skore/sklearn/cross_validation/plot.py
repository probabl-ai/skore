"""Plot cross-validation results."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects


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
    from datetime import timedelta

    import pandas
    import plotly
    import plotly.graph_objects as go

    _cv_results = cv_results.copy()

    with contextlib.suppress(KeyError):
        del _cv_results["indices"]
        del _cv_results["estimator"]

    df = pandas.DataFrame(_cv_results)

    # Move time columns to last and "test_score" to first
    if "fit_time" in df.columns:
        df.insert(len(df.columns) - 1, "fit_time", df.pop("fit_time"))
    if "score_time" in df.columns:
        df.insert(len(df.columns) - 1, "score_time", df.pop("score_time"))
    if "test_score" in df.columns:
        df.insert(0, "test_score", df.pop("test_score"))

    dict_labels = {
        "fit_time": "fit_time (seconds)",
        "score_time": "score_time (seconds)",
        "fit_time_per_data_point": "fit_time_per_data_point (seconds)",
        "score_time_per_data_point": "score_time_per_data_point (seconds)",
    }

    def linspace(lo, hi, num):
        interval = (hi - lo) / num
        return [lo + k * interval for k in range(0, num + 1)]

    fig = go.Figure()

    for col_i, col_name in enumerate(df.columns):
        metric_name = dict_labels.get(col_name, col_name)
        bar_color = plotly.colors.qualitative.Plotly[
            col_i % len(plotly.colors.qualitative.Plotly)
        ]
        bar_x = linspace(min(df.index) - 0.5, max(df.index) + 0.5, num=10)

        common_kwargs = dict(
            visible=True if col_i == 0 else "legendonly",
            legendgroup=f"group{col_i}",
            # If the metric is a duration (e.g. "fit_time"),
            # we show a different hover text
            hovertemplate=(
                "%{customdata}" f"<extra>{col_name} (timedelta)</extra>"
                if ("fit_time" in col_name or "score_time" in col_name)
                else "%{y}"
            ),
            customdata=(
                [str(timedelta(seconds=x)) for x in df[col_name].values]
                if ("fit_time" in col_name or "score_time" in col_name)
                else None
            ),
        )

        # Calculate statistics
        avg_value = df[col_name].mean()
        std_value = df[col_name].std()

        # Add all traces at once
        fig.add_traces(
            [
                # Bar trace
                go.Bar(
                    x=df.index,
                    y=df[col_name].values,
                    name=metric_name,
                    marker_color=bar_color,
                    showlegend=True,
                    **common_kwargs,
                ),
                # Mean line
                go.Scatter(
                    x=bar_x,
                    y=[avg_value] * 10,
                    name=f"Average {metric_name}",
                    line=dict(dash="dash", color=bar_color),
                    showlegend=False,
                    mode="lines",
                    **common_kwargs,
                ),
                # +1 std line
                go.Scatter(
                    x=bar_x,
                    y=[avg_value + std_value] * 10,
                    name=f"Average + 1 std. dev. {metric_name}",
                    line=dict(dash="dot", color=bar_color),
                    showlegend=False,
                    mode="lines",
                    **common_kwargs,
                ),
                # -1 std line
                go.Scatter(
                    x=bar_x,
                    y=[avg_value - std_value] * 10,
                    name=f"Average - 1 std. dev. {metric_name}",
                    line=dict(dash="dot", color=bar_color),
                    showlegend=False,
                    mode="lines",
                    **common_kwargs,
                ),
            ]
        )

    fig.update_xaxes(tickmode="linear", dtick=1, title_text="Split number")
    fig.update_yaxes(title_text="Value")
    fig.update_layout(title_text="Cross-validation results for each split")

    return fig
