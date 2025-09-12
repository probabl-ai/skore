from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from sklearn.utils.validation import (
    _check_pos_label_consistency,
    check_consistent_length,
)

from skore._sklearn.types import (
    MLTask,
    PositiveLabel,
    ReportType,
    YPlotData,
)

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


class _ClassifierCurveDisplayMixin:
    """Mixin class to be used in Displays requiring a binary classifier.

    The aim of this class is to centralize some validations regarding the estimator and
    the target and gather the response of the estimator.
    """

    # defined in the concrete display class
    estimator_name: str
    ml_task: MLTask
    pos_label: PositiveLabel | None

    def _validate_curve_kwargs(
        self,
        *,
        curve_param_name: str,
        curve_kwargs: dict[str, Any] | list[dict[str, Any]] | None,
        n_curves: int,
        report_type: ReportType,
    ) -> list[dict[str, Any]]:
        """Validate and format the classification curve keyword arguments.

        Parameters
        ----------
        curve_param_name : str
            The name of the curve parameter.

        curve_kwargs : dict or list of dict or None
            Keyword arguments to customize the classification curve.

        n_curves : int
            The number of curves we are plotting.

        report_type : {"comparison-cross-validation", "comparison-estimator",
                      "cross-validation", "estimator"}
            The type of report.

        Returns
        -------
        list of dict
            Validated list of keyword arguments for each curve.

        Raises
        ------
        ValueError
            If the format of curve_kwargs is invalid.
        """
        if self.ml_task == "binary-classification":
            if report_type in ("estimator", "cross-validation"):
                allow_single_dict = True
            elif report_type in ("comparison-estimator", "comparison-cross-validation"):
                # since we compare different estimators, it does not make sense to share
                # a single dictionary for all the estimators.
                allow_single_dict = False
            else:
                raise ValueError(
                    f"`report_type` should be one of 'estimator', 'cross-validation', "
                    "'comparison-cross-validation' or 'comparison-estimator'. "
                    f"Got '{report_type}' instead."
                )
        else:
            allow_single_dict = False

        if curve_kwargs is None:
            return [{}] * n_curves
        elif n_curves == 1:
            if isinstance(curve_kwargs, dict):
                return [curve_kwargs]
            elif isinstance(curve_kwargs, list) and len(curve_kwargs) == 1:
                return curve_kwargs
            else:
                raise ValueError(
                    "You intend to plot a single curve. We expect "
                    f"`{curve_param_name}` to be a dictionary. Got "
                    f"{type(curve_kwargs)} instead."
                )
        else:  # n_curves > 1
            if not allow_single_dict and isinstance(curve_kwargs, dict):
                raise ValueError(
                    "You intend to plot multiple curves. We expect "
                    f"`{curve_param_name}` to be a list of dictionaries. Got "
                    f"{type(curve_kwargs)} instead."
                )
            if isinstance(curve_kwargs, dict):
                return [curve_kwargs] * n_curves

            # From this point, we have a list of dictionaries
            if len(curve_kwargs) != n_curves:
                raise ValueError(
                    "You intend to plot multiple curves. We expect "
                    f"`{curve_param_name}` to be a list of dictionaries with the "
                    f"same length as the number of curves. Got "
                    f"{len(curve_kwargs)} instead of {n_curves}."
                )

            return curve_kwargs

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
    x_range: tuple[float, float] = (0, 1),
    y_range: tuple[float, float] = (0, 1),
    offset: float = 10,
) -> None:
    """Despine the matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axis to despine.
    x_range : tuple of float, default=(0, 1)
        The range of the x-axis.
    y_range : tuple of float, default=(0, 1)
        The range of the y-axis.
    offset : float, default=10
        The offset to add to the x-axis and y-axis.
    """
    ax.spines["bottom"].set_bounds(x_range[0], x_range[1])
    ax.spines["left"].set_bounds(y_range[0], y_range[1])
    ax.spines["left"].set_position(("outward", offset))
    ax.spines["bottom"].set_position(("outward", offset))
    for s in axis_to_despine:
        ax.spines[s].set_visible(False)


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
