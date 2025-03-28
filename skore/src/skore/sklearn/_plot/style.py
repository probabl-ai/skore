from functools import wraps
from typing import Any, Callable

import seaborn as sns


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
        1. Sets seaborn's plotting context to "talk"
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
            with sns.plotting_context("notebook"):
                result = plot_func(self, *args, **kwargs)
                if hasattr(self, "figure_"):
                    self.figure_.tight_layout()
                return result

        return wrapper
