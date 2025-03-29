from functools import wraps
from typing import Any, Callable

import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_STYLE = {
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
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

    def set_style(self, **kwargs: Any):
        """Set the style parameters for the display.

        Parameters
        ----------
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
            setattr(self, default_attr, param_value)
        return self

    @staticmethod
    def style_plot(plot_func: Callable) -> Callable:
        """Apply consistent style to skore displays.

        This decorator:
        1. Applies default style settings
        2. Executes the plotting code
        3. Applies `tight_layout` to the figure if it exists

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
            is_interactive = plt.isinteractive()
            with sns.plotting_context(DEFAULT_STYLE):
                result = plot_func(self, *args, **kwargs)
                self.figure_.tight_layout()
            if is_interactive:
                # the context manager from matplotlib will reset the interactive mode to
                # the default state, before to the execution of a first plot. Therefore,
                # it can be falsely set to non-interactive mode. We can restore the
                # state by explicitly calling `plt.ion()`
                #
                # See: https://github.com/matplotlib/matplotlib/issues/26716
                plt.ion()
            return result

        return wrapper
