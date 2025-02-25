from typing import Any


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
