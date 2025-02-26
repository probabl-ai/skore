import inspect
from collections.abc import Sequence
from io import StringIO
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure, SubFigure
from numpy.typing import ArrayLike
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from sklearn.utils.validation import (
    _check_pos_label_consistency,
    check_consistent_length,
)


class HelpDisplayMixin:
    """Mixin class to add help functionality to a class."""

    estimator_name: str  # defined in the concrete display class

    def _get_attributes_for_help(self) -> list[str]:
        """Get the attributes ending with '_' to display in help."""
        attributes = []
        for name in dir(self):
            if name.endswith("_") and not name.startswith("_"):
                attributes.append(f".{name}")
        return sorted(attributes)

    def _get_methods_for_help(self) -> list[tuple[str, Any]]:
        """Get the public methods to display in help."""
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        filtered_methods = []
        for name, method in methods:
            is_private = name.startswith("_")
            is_class_method = inspect.ismethod(method) and method.__self__ is type(self)
            is_help_method = name == "help"
            if not (is_private or is_class_method or is_help_method):
                filtered_methods.append((f".{name}(...)", method))
        return sorted(filtered_methods)

    def _create_help_tree(self) -> Tree:
        """Create a rich Tree with attributes and methods."""
        tree = Tree("display")

        attributes = self._get_attributes_for_help()
        attr_branch = tree.add("[bold cyan] Attributes[/bold cyan]")
        # Ensure figure_ and ax_ are first
        sorted_attrs = sorted(attributes)
        if ("figure_" in sorted_attrs) and ("ax_" in sorted_attrs):
            sorted_attrs.remove(".ax_")
            sorted_attrs.remove(".figure_")
            sorted_attrs = [".figure_", ".ax_"] + [
                attr for attr in sorted_attrs if attr not in [".figure_", ".ax_"]
            ]
        for attr in sorted_attrs:
            attr_branch.add(attr)

        methods = self._get_methods_for_help()
        method_branch = tree.add("[bold cyan]Methods[/bold cyan]")
        for name, method in methods:
            description = (
                method.__doc__.split("\n")[0]
                if method.__doc__
                else "No description available"
            )
            method_branch.add(f"{name} - {description}")

        return tree

    def _create_help_panel(self) -> Panel:
        return Panel(
            self._create_help_tree(),
            title=(
                f"[bold cyan]{self.__class__.__name__} for "
                f"{self.estimator_name}[/bold cyan]"
            ),
            border_style="orange1",
            expand=False,
        )

    def help(self) -> None:
        """Display available attributes and methods using rich."""
        from skore import console  # avoid circular import

        console.print(self._create_help_panel())

    def __str__(self) -> str:
        """Return a string representation using rich."""
        string_buffer = StringIO()
        console = Console(file=string_buffer, force_terminal=False)
        console.print(
            Panel(
                "Get guidance using the display.help() method",
                title=f"[cyan]{self.__class__.__name__}[/cyan]",
                border_style="orange1",
                expand=False,
            )
        )
        return string_buffer.getvalue()

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        string_buffer = StringIO()
        console = Console(file=string_buffer, force_terminal=False)
        console.print(f"[cyan]skore.{self.__class__.__name__}(...)[/cyan]")
        return string_buffer.getvalue()


class _ClassifierCurveDisplayMixin:
    """Mixin class to be used in Displays requiring a binary classifier.

    The aim of this class is to centralize some validations regarding the estimator and
    the target and gather the response of the estimator.
    """

    estimator_name: str  # defined in the concrete display class

    def _validate_plot_params(
        self, *, ax: Optional[Axes] = None, estimator_name: Optional[str] = None
    ) -> tuple[Axes, Union[Figure, SubFigure], str]:
        if ax is None:
            _, ax = plt.subplots()

        estimator_name = (
            self.estimator_name if estimator_name is None else estimator_name
        )
        return ax, ax.figure, estimator_name

    @classmethod
    def _validate_from_predictions_params(
        cls,
        y_true: Sequence[ArrayLike],
        y_pred: Sequence[ArrayLike],
        *,
        ml_task: str,
        pos_label: Optional[Union[int, float, bool, str]] = None,
    ) -> Union[int, float, bool, str, None]:
        for y_true_i, y_pred_i in zip(y_true, y_pred):
            check_consistent_length(y_true_i, y_pred_i)

        if ml_task == "binary-classification":
            pos_label = _check_pos_label_consistency(pos_label, y_true[0])

        return pos_label


def _despine_matplotlib_axis(
    ax: Axes,
    *,
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
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_bounds(x_range[0], x_range[1])
    ax.spines["left"].set_bounds(y_range[0], y_range[1])
    ax.spines["left"].set_position(("outward", offset))
    ax.spines["bottom"].set_position(("outward", offset))


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
