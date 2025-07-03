from collections.abc import Callable
from functools import wraps
from typing import Any

import matplotlib.pyplot as plt

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
        2. Executes `plot_func`
        3. Applies `tight_layout`

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
                plt.rcParams.update(original_params)
            return result

        return wrapper
