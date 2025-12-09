import inspect
from collections.abc import Callable
from functools import wraps
from io import StringIO
from typing import Any, Literal, Protocol, runtime_checkable

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from skore._config import get_config
from skore._sklearn.types import PlotBackend

########################################################################################
# Display protocol
########################################################################################


@runtime_checkable
class Display(Protocol):
    """Protocol specifying the common API for all `skore` displays.

    .. note::
       This class is a Python protocol and it is not intended to be inherited from.
    """

    def plot(self, **kwargs: Any) -> None:
        """Display a figure containing the information of the display."""

    def set_style(self, **kwargs: Any) -> None:
        """Set the style of the display."""

    def frame(self, **kwargs: Any) -> pd.DataFrame:
        """Get the data used to create the display.

        Returns
        -------
        DataFrame
            A DataFrame containing the data used to create the display.
        """

    def help(self) -> None:
        """Display available attributes and methods using rich."""


########################################################################################
# Plotting related mixins
########################################################################################


class PlotBackendMixin:
    """Mixin class for Displays to dispatch plotting to the configured backend."""

    def _plot(self, **kwargs):
        """Dispatch plotting to the configured backend."""
        plot_backend = get_config()["plot_backend"]
        if plot_backend == "matplotlib":
            return self._plot_matplotlib(**kwargs)
        elif plot_backend == "plotly":
            return self._plot_plotly(**kwargs)
        else:
            raise NotImplementedError(
                f"Plotting backend {plot_backend} not available. "
                f"Available options are {PlotBackend.__args__}."
            )

    def _plot_plotly(self, **kwargs):
        raise NotImplementedError(
            "Plotting with plotly is not supported for this Display."
        )


DEFAULT_STYLE = {
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "axes.linewidth": 1.25,
    "grid.linewidth": 1.25,
    "lines.linewidth": 1.75,
    "lines.markersize": 6,
    "patch.linewidth": 1.25,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.minor.width": 1.25,
    "ytick.minor.width": 1.25,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.minor.size": 5,
    "ytick.minor.size": 5,
    "legend.loc": "upper left",
    "legend.borderaxespad": 0,
}


class StyleDisplayMixin:
    """Mixin to control the style plot of a display."""

    @property
    def _style_params(self) -> list[str]:
        """Get the list of available style parameters.

        Returns
        -------
        list
            List of style parameter names (without '_default_' prefix).
        """
        prefix = "_default_"
        suffix = "_kwargs"
        return [
            attr[len(prefix) :]
            for attr in dir(self)
            if attr.startswith(prefix) and attr.endswith(suffix)
        ]

    def set_style(
        self, *, policy: Literal["override", "update"] = "override", **kwargs: Any
    ):
        """Set the style parameters for the display.

        Parameters
        ----------
        policy : Literal["override", "update"], default="override"
            Policy to use when setting the style parameters.
            If "override", existing settings are set to the provided values.
            If "update", existing settings are not changed; only settings that were
            previously unset are changed.

        **kwargs : dict
            Style parameters to set. Each parameter name should correspond to a
            a style attribute passed to the plot method of the display.

        Returns
        -------
        self : object
            Returns the instance itself.

        Raises
        ------
        ValueError
            If a style parameter is unknown.
        """
        for param_name, param_value in kwargs.items():
            default_attr = f"_default_{param_name}"
            if not hasattr(self, default_attr):
                raise ValueError(
                    f"Unknown style parameter: {param_name}. "
                    f"The parameter name should be one of {self._style_params}."
                )
            if policy == "override":
                setattr(self, default_attr, param_value)
            elif policy == "update":
                current_value = getattr(self, default_attr)
                if current_value is None:
                    setattr(self, default_attr, param_value)
                else:
                    setattr(self, default_attr, {**current_value, **param_value})
            else:
                raise ValueError(
                    f"Invalid policy: {policy}. "
                    f"Valid policies are 'override' and 'update'."
                )
        return self

    @staticmethod
    def style_plot(plot_func: Callable) -> Callable:
        """Apply consistent style to skore displays.

        This decorator:
        1. Applies default style settings
        2. Executes `plot_func`
        3. Calls `plt.tight_layout()` to make sure axis does not overlap
        4. Restores the original style settings

        Parameters
        ----------
        plot_func : callable
            The plot function to be decorated.

        Returns
        -------
        callable
            The decorated plot function.
        """

        @wraps(plot_func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            # We need to manually handle setting the style of the parameters because
            # `plt.style.context` has a side effect with the interactive mode.
            # See https://github.com/matplotlib/matplotlib/issues/25041
            original_params = {key: plt.rcParams[key] for key in DEFAULT_STYLE}
            plt.rcParams.update(DEFAULT_STYLE)
            try:
                result = plot_func(self, *args, **kwargs)
            finally:
                plt.tight_layout()
                plt.rcParams.update(original_params)
            return result

        return wrapper


########################################################################################
# General purpose mixins
########################################################################################


class HelpDisplayMixin:
    """Mixin class to add help functionality to a class."""

    estimator_name: str  # defined in the concrete display class

    def _get_attributes_for_help(self) -> list[str]:
        """Get the attributes ending with '_' to display in help."""
        return sorted(
            f".{name}"
            for name in dir(self)
            if name.endswith("_") and not name.startswith("_")
        )

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
            title=f"[bold cyan]{self.__class__.__name__} [/bold cyan]",
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
                "Get guidance using the help() method",
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


########################################################################################
# Display mixin inheriting from the different mixins
########################################################################################


class DisplayMixin(HelpDisplayMixin, PlotBackendMixin, StyleDisplayMixin):
    """Mixin inheriting help, plotting, and style functionality."""
