"""Plot cross-validation results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import linspace

if TYPE_CHECKING:
    import plotly.graph_objects


def plot_cross_validation_compare_scores(
    cv_results: dict,
) -> plotly.graph_objects.Figure:
    """Plot the result of a cross-validation run.

    Parameters
    ----------
    cv_results : dict
        The output of scikit-learn's cross_validate function.

    Returns
    -------
    plotly.graph_objects.Figure
        A plot of the cross-validation results, without the time-related results.
    """
    import pandas
    import plotly
    import plotly.graph_objects as go

    _cv_results = cv_results.copy()

    # Remove irrelevant keys
    to_remove = ["indices", "estimator"] + [
        key
        for key in _cv_results
        if key.startswith("fit_time") or key.startswith("score_time")
    ]
    for key in to_remove:
        _cv_results.pop(key, None)

    df = pandas.DataFrame(_cv_results)

    if "test_score" in df.columns:
        df.insert(0, "test_score", df.pop("test_score"))

    fig = go.Figure()

    for col_i, col_name in enumerate(df.columns):
        metric_name = col_name
        bar_color = plotly.colors.qualitative.Plotly[
            col_i % len(plotly.colors.qualitative.Plotly)
        ]
        bar_x = linspace(min(df.index) - 0.5, max(df.index) + 0.5, num=10)

        common_kwargs = dict(
            visible=True if col_i == 0 else "legendonly",
            legendgroup=f"group{col_i}",
            hovertemplate="%{y}",
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

    fig.update_xaxes(tickmode="linear", dtick=1, title_text="Split index")
    fig.update_yaxes(title_text="Value")
    fig.update_layout(title_text="Cross-validation results for each split")

    return fig
