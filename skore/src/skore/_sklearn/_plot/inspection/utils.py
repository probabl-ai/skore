from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from skore._sklearn._plot.utils import _despine_matplotlib_axis


def select_k_features_in_group(
    frame: pd.DataFrame,
    select_k: int,
    *,
    score_raw_column: str,
    use_absolute: bool = False,
) -> pd.DataFrame:
    values_grouped = frame.groupby("feature")[score_raw_column]
    scores = (
        values_grouped.apply(lambda x: x.abs().mean())
        if use_absolute
        else values_grouped.mean()
    )
    if select_k > 0:
        selected_features = scores.nlargest(abs(select_k)).index
    else:
        selected_features = scores.nsmallest(abs(select_k)).index
    return frame[frame["feature"].isin(selected_features)]


def sort_features_in_group(
    frame: pd.DataFrame,
    sorting_order: Literal["descending", "ascending"],
    *,
    score_raw_column: str,
    use_absolute: bool = False,
) -> pd.DataFrame:
    ascending = sorting_order == "ascending"
    values_grouped = frame.groupby("feature")[score_raw_column]
    scores = (
        values_grouped.apply(lambda x: x.abs().mean())
        if use_absolute
        else values_grouped.mean()
    )
    feature_order = scores.sort_values(ascending=ascending).index
    return frame.set_index("feature").loc[feature_order].reset_index()


def select_k_features(
    frame: pd.DataFrame,
    select_k: int,
    group_columns: list[str],
    *,
    score_raw_column: str,
    use_absolute: bool = False,
) -> pd.DataFrame:
    if not group_columns:
        return select_k_features_in_group(
            frame,
            select_k,
            score_raw_column=score_raw_column,
            use_absolute=use_absolute,
        )
    return pd.concat(
        [
            select_k_features_in_group(
                group,
                select_k,
                score_raw_column=score_raw_column,
                use_absolute=use_absolute,
            )
            for _, group in frame.groupby(group_columns, observed=True)
        ],
        ignore_index=True,
    )


def sort_features(
    frame: pd.DataFrame,
    sorting_order: Literal["descending", "ascending"],
    group_columns: list[str],
    *,
    score_raw_column: str,
    use_absolute: bool = False,
) -> pd.DataFrame:
    if not group_columns:
        return sort_features_in_group(
            frame,
            sorting_order,
            score_raw_column=score_raw_column,
            use_absolute=use_absolute,
        )
    return pd.concat(
        [
            sort_features_in_group(
                group,
                sorting_order,
                score_raw_column=score_raw_column,
                use_absolute=use_absolute,
            )
            for _, group in frame.groupby(group_columns, sort=False, observed=True)
        ],
        ignore_index=True,
    )


def _decorate_matplotlib_axis(
    *,
    ax: plt.Axes,
    add_background_features: bool,
    n_features: int,
    xlabel: str,
    ylabel: str,
) -> None:
    """Decorate the matplotlib axis.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axis to decorate.
    add_background_features : bool
        Whether to add a background color for each group of features.
    n_features : int
        The number of features to displayed.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    """
    ax.axvline(x=0, color=".5", linestyle="--")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    _despine_matplotlib_axis(
        ax,
        axis_to_despine=("top", "right", "left"),
        remove_ticks=True,
        x_range=None,
        y_range=None,
    )
    if add_background_features:
        for feature_idx in range(0, n_features, 2):
            ax.axhspan(
                feature_idx - 0.5,
                feature_idx + 0.5,
                color="lightgray",
                alpha=0.4,
                zorder=0,
            )
