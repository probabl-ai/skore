from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from pandas import DataFrame
from sklearn.utils.validation import (
    _check_pos_label_consistency,
    check_consistent_length,
)

from skore._sklearn._base import _get_cached_response_values
from skore._sklearn.types import (
    DataSource,
    MLTask,
    PositiveLabel,
    ReportType,
    YPlotData,
)

if TYPE_CHECKING:
    from skore._sklearn._base import Cache

LINESTYLE = [
    ("solid", "solid"),
    ("dotted", "dotted"),
    ("dashed", "dashed"),
    ("dashdot", "dashdot"),
    ("loosely dotted", (0, (1, 10))),
    ("dotted", (0, (1, 5))),
    ("densely dotted", (0, (1, 1))),
    ("long dash with offset", (5, (10, 3))),
    ("loosely dashed", (0, (5, 10))),
    ("dashed", (0, (5, 5))),
    ("densely dashed", (0, (5, 1))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]


class _ClassifierDisplayMixin:
    """Mixin class to be used in Displays requiring a classifier."""

    # defined in the concrete display class
    estimator_name: str
    ml_task: MLTask
    pos_label: PositiveLabel | None

    @classmethod
    def _validate_from_predictions_params(
        cls,
        y_true: Sequence[YPlotData],
        y_pred: Sequence[YPlotData],
        *,
        ml_task: str,
        pos_label: PositiveLabel | None = None,
    ) -> PositiveLabel | None:
        for y_true_i, y_pred_i in zip(y_true, y_pred, strict=False):
            check_consistent_length(y_true_i.y, y_pred_i.y)

        if ml_task == "binary-classification":
            pos_label = _check_pos_label_consistency(pos_label, y_true[0].y)

        return pos_label


def _rotate_ticklabels(
    ax: Axes, *, rotation: float = 45, horizontalalignment: str = "right"
) -> None:
    """Rotate the tick labels of the axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to rotate.
    rotation : float, default=45
        The rotation of the tick labels.
    horizontalalignment : {"left", "center", "right"}, default="right"
        The horizontal alignment of the tick labels.
    """
    plt.setp(
        ax.get_xticklabels(), rotation=rotation, horizontalalignment=horizontalalignment
    )


def _get_adjusted_fig_size(
    fig: Figure, ax: Axes, direction: str, target_size: float
) -> float:
    """Get the adjusted figure size.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to adjust.

    ax : matplotlib.axes.Axes
        The axis to adjust.

    direction : {"width", "height"}
        The direction to adjust.

    target_size : float
        The target size.

    Returns
    -------
    float
        The adjusted figure size.
    """
    size_display = getattr(ax.get_window_extent(), direction)
    size = fig.dpi_scale_trans.inverted().transform((size_display, 0))[0]
    dim = 0 if direction == "width" else 1
    fig_size = fig.get_size_inches()[dim]
    return target_size * (fig_size / size)


def _adjust_fig_size(
    fig: Figure, ax: Axes, target_width: float, target_height: float
) -> None:
    """Rescale a figure to the target width and height.

    This allows us to have all figures of a given type (bar plots, lines or
    histograms) have the same size, so that the displayed report looks more
    uniform, without having to do manual adjustments to account for the length
    of labels, occurrence of titles etc. We let pyplot generate the figure
    without any size constraints then resize it and thus let matplotlib deal
    with resizing the appropriate elements (eg shorter bars when the labels
    take more horizontal space).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to adjust.

    ax : matplotlib.axes.Axes
        The axis to adjust.

    target_width : float
        The target width.

    target_height : float
        The target height.
    """
    width = _get_adjusted_fig_size(fig, ax, "width", target_width)
    height = _get_adjusted_fig_size(fig, ax, "height", target_height)
    fig.set_size_inches((width, height))


def _despine_matplotlib_axis(
    ax: Axes,
    *,
    axis_to_despine: tuple = ("top", "right"),
    remove_ticks: bool = False,
    x_range: tuple[float, float] | None = (0, 1),
    y_range: tuple[float, float] | None = (0, 1),
    offset: float = 10,
) -> None:
    """Despine the matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axis to despine.
    axis_to_despine : tuple of str, default=("top", "right")
        The axes to despine.
    remove_ticks : bool, default=False
        Whether to remove the ticks when adding "left" and "bottom" to the axis to
        despine.
    x_range : tuple of float, default=(0, 1)
        The range of the x-axis.
    y_range : tuple of float, default=(0, 1)
        The range of the y-axis.
    offset : float, default=10
        The offset to add to the x-axis and y-axis.
    """
    if x_range is not None:
        ax.spines["bottom"].set_bounds(x_range[0], x_range[1])
    if y_range is not None:
        ax.spines["left"].set_bounds(y_range[0], y_range[1])
    ax.spines["left"].set_position(("outward", offset))
    ax.spines["bottom"].set_position(("outward", offset))
    for s in axis_to_despine:
        ax.spines[s].set_visible(False)
        if remove_ticks and s in ("left", "bottom"):
            axis: Literal["x", "y"] = "x" if s == "bottom" else "y"
            ax.tick_params(axis=axis, length=0)


def _validate_style_kwargs(
    default_style_kwargs: dict[str, Any], user_style_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Create valid style kwargs by avoiding Matplotlib alias errors.

    Matplotlib raises an error when, for example, 'color' and 'c', or 'linestyle' and
    'ls', are specified together. To avoid this, we automatically keep only the one
    specified by the user and raise an error if the user specifies both.

    Parameters
    ----------
    default_style_kwargs : dict
        The Matplotlib style kwargs used by default in the scikit-learn display.
    user_style_kwargs : dict
        The user-defined Matplotlib style kwargs.

    Returns
    -------
    valid_style_kwargs : dict
        The validated style kwargs taking into account both default and user-defined
        Matplotlib style kwargs.
    """
    invalid_to_valid_kw = {
        "ls": "linestyle",
        "c": "color",
        "ec": "edgecolor",
        "fc": "facecolor",
        "lw": "linewidth",
        "mec": "markeredgecolor",
        "mfcalt": "markerfacecoloralt",
        "ms": "markersize",
        "mew": "markeredgewidth",
        "mfc": "markerfacecolor",
        "aa": "antialiased",
        "ds": "drawstyle",
        "font": "fontproperties",
        "family": "fontfamily",
        "name": "fontname",
        "size": "fontsize",
        "stretch": "fontstretch",
        "style": "fontstyle",
        "variant": "fontvariant",
        "weight": "fontweight",
        "ha": "horizontalalignment",
        "va": "verticalalignment",
        "ma": "multialignment",
    }
    for invalid_key, valid_key in invalid_to_valid_kw.items():
        if invalid_key in user_style_kwargs and valid_key in user_style_kwargs:
            raise TypeError(
                f"Got both {invalid_key} and {valid_key}, which are aliases of one "
                "another"
            )
    valid_style_kwargs = default_style_kwargs.copy()

    for key in user_style_kwargs:
        if key in invalid_to_valid_kw:
            valid_style_kwargs[invalid_to_valid_kw[key]] = user_style_kwargs[key]
        else:
            valid_style_kwargs[key] = user_style_kwargs[key]

    return valid_style_kwargs


def sample_mpl_colormap(
    cmap: Colormap, n: int
) -> list[tuple[float, float, float, float]]:
    """Sample colors from a Matplotlib colormap.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        The Matplotlib colormap to sample from.
    n : int
        The number of colors to sample.

    Returns
    -------
    colors : list of str
        The sampled colors.
    """
    indices = np.linspace(0, 1, n)
    return [cmap(i) for i in indices]


def _get_curve_plot_columns(
    plot_data: DataFrame,
    report_type: ReportType,
    ml_task: MLTask,
    data_source: DataSource | Literal["both"],
    subplot_by: Literal["auto", "label", "estimator", "data_source"] | None = "auto",
) -> tuple[str | None, str | None, str | None]:
    """Determine col, hue, and style columns for precision-recall and ROC.

    Rules:
    - Default ("auto"): None for EstimatorReport and Cross-Validation Report,
        "estimator" for ComparisonReport
    - subplot_by=None disallowed for comparison in multiclass
    - subplot_by="estimator" only allowed for comparison reports
    - subplot_by="label" only allowed for multiclass classification
    - subplot_by="data_source" only allowed for EstimatorReport with both data \
        sources
    - hue priority: estimator > label > data_source (excluding col)

    Returns (col, hue, style) tuple where each can be None if not applicable.
    """
    has_multiple_estimators = (
        "estimator" in plot_data.columns and plot_data["estimator"].nunique() > 1
    )
    is_comparison = "comparison" in report_type
    is_multiclass = ml_task == "multiclass-classification"
    has_both_data_sources = data_source == "both"

    allowed_values: set[str | None] = {"auto"}
    if is_multiclass:
        allowed_values.add("label")
        if not is_comparison:
            allowed_values.add(None)
    else:
        allowed_values.add(None)
    if is_comparison and has_multiple_estimators:
        allowed_values.add("estimator")
    if has_both_data_sources and (not is_comparison or not is_multiclass):
        allowed_values.add("data_source")
    # Disallow for comparison reports in multiclass classification

    if subplot_by not in allowed_values:
        allowed_str = ", ".join(sorted([str(s) for s in allowed_values]))
        raise ValueError(
            f"subplot_by must be one of {allowed_str}. Got {subplot_by!r} instead."
        )

    if subplot_by == "auto":
        col = "estimator" if is_comparison else None
    else:
        col = subplot_by
    has_multiple_labels = (
        "label" in plot_data.columns and plot_data["label"].nunique() > 1
    )

    hue_candidates = []
    if has_multiple_estimators:
        hue_candidates.append("estimator")
    if has_multiple_labels:
        hue_candidates.append("label")

    hue = hue[0] if (hue := [c for c in hue_candidates if c != col]) else None

    style = "data_source" if has_both_data_sources else None

    return col, hue, style


def _build_custom_legend_with_stats(
    *,
    ax: Axes,
    subplot_data: DataFrame,
    hue: str | None,
    style: str | None,
    hue_order: list[Any] | None,
    style_order: list[Any] | None,
    is_cross_validation: bool = False,
    statistic_column_name: Literal["average_precision", "roc_auc"],
    statistic_acronym: Literal["AP", "AUC"],
    chance_level_label: str | None = None,
) -> None:
    """Build custom legend with a custom statistic for a single axis."""
    legend_labels = []
    for hue_value in hue_order or [None]:
        hue_value_str = (
            f"'{hue_value}'" if isinstance(hue_value, str) else str(hue_value)
        )
        hue_group = (
            subplot_data.query(f"{hue} == {hue_value_str}")
            if hue_value is not None
            else subplot_data
        )
        for style_value in style_order or [None]:
            style_value_str = (
                f"'{style_value}'" if isinstance(style_value, str) else str(style_value)
            )
            statistic_group = (
                hue_group.query(f"{style} == {style_value_str}")[statistic_column_name]
                if style_value is not None
                else hue_group[statistic_column_name]
            )
            if not statistic_group.empty:
                statistic = (
                    f"{statistic_acronym}={statistic_group.mean():.2f}±{statistic_group.std():.2f}"
                    if is_cross_validation and len(statistic_group) > 1
                    else f"{statistic_acronym}={statistic_group.iloc[0]:.2f}"
                )
                legend_labels.append(
                    _format_legend_label(
                        style_column_name=style,
                        style_value=style_value,
                        hue_value=hue_value,
                        statistic=statistic,
                    )
                )
    if chance_level_label is not None:
        legend_labels.append(chance_level_label)

    n_entries = len(legend_labels)
    lines = ax.get_lines()
    handles = []
    seen_line_attributes = []
    for line in lines:
        line_attributes = (line.get_color(), line.get_linestyle())
        if line_attributes not in seen_line_attributes:
            seen_line_attributes.append(line_attributes)
            handles.append(line)

    fontsize = "small" if n_entries > 4 else "medium"

    ax.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=1,
        frameon=False,
        fontsize=fontsize,
    )


def _format_legend_label(
    style_column_name: str | None,
    style_value: str | None,
    hue_value: str | None,
    statistic: str,
) -> str:
    """Format a legend label based on style and hue."""
    if style_value is None and hue_value is None:
        return statistic
    if style_column_name == "data_source":
        style_value = cast(str, style_value)
        style_value = style_value.title() + " set"
    return (
        " - ".join(
            str(s) for s in filter(lambda x: x is not None, [hue_value, style_value])
        )
        + f" ({statistic})"
    )


def _expand_data_sources(
    data_source: DataSource | Literal["both"],
) -> tuple[DataSource, ...]:
    """Expand 'both' data source to ('train', 'test')."""
    if data_source == "both":
        return ("train", "test")
    return (data_source,)


def _get_ys_for_single_report(
    *,
    cache: "Cache",
    estimator_hash: int,
    estimator: Any,
    estimator_name: str,
    X: ArrayLike | None,
    y_true: ArrayLike,
    data_source: DataSource,
    data_source_hash: int | None,
    response_method: str | list[str] | tuple[str, ...],
    pos_label: PositiveLabel | None,
    split: int | None = None,
) -> tuple[YPlotData, YPlotData]:
    """Get y_true and y_pred as YPlotData for a single estimator report."""
    results = _get_cached_response_values(
        cache=cache,
        estimator_hash=estimator_hash,
        estimator=estimator,
        X=X,
        response_method=response_method,
        pos_label=pos_label,
        data_source=data_source,
        data_source_hash=data_source_hash,
    )

    for key, value, is_cached in results:
        if not is_cached:
            cache[key] = value
        if key[-1] != "predict_time":
            y_pred = value

    y_true_data = YPlotData(
        estimator_name=estimator_name,
        data_source=data_source,
        split=split,
        y=y_true,
    )
    y_pred_data = YPlotData(
        estimator_name=estimator_name,
        data_source=data_source,
        split=split,
        y=y_pred,
    )

    return y_true_data, y_pred_data
