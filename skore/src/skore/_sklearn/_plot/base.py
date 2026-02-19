from collections.abc import Callable
from functools import wraps
from typing import Any, Literal, Protocol, runtime_checkable

import matplotlib.pyplot as plt
import pandas as pd

from skore._config import get_config
from skore._sklearn.types import PlotBackend
from skore._utils.repr.base import DisplayHelpMixin

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

    def set_style(
        self, *, policy: Literal["override", "update"] = "update", **kwargs: Any
    ) -> None:
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
    "boxplot.patchartist": True,
    "boxplot.boxprops.color": "black",
    "boxplot.capprops.color": "black",
    "boxplot.medianprops.color": "black",
    "boxplot.whiskerprops.color": "black",
    "boxplot.boxprops.linewidth": 1.0,
    "boxplot.capprops.linewidth": 1.0,
    "boxplot.medianprops.linewidth": 0.5,
    "boxplot.whiskerprops.linewidth": 1.0,
}

BOXPLOT_STYLE = {
    "patch_artist": True,
    "boxprops": {"edgecolor": "black", "facecolor": "none"},
    "whiskerprops": {"color": "black"},
    "capprops": {"color": "black"},
    "medianprops": {"color": "black"},
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
        self, *, policy: Literal["override", "update"] = "update", **kwargs: Any
    ):
        """Set the style parameters for the display.

        Parameters
        ----------
        policy : Literal["override", "update"], default="update"
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
            The instance with a modified style.

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
                if hasattr(self, "facet_"):
                    self.facet_.tight_layout()
                else:
                    plt.tight_layout()
                plt.rcParams.update(original_params)
            return result

        return wrapper


########################################################################################
# Display mixin inheriting from the different mixins
########################################################################################


class DisplayMixin(DisplayHelpMixin, PlotBackendMixin, StyleDisplayMixin):
    """Mixin inheriting help, plotting, and style functionality."""
