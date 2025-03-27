from typing import Any, Optional, Protocol, runtime_checkable

from matplotlib.axes import Axes


@runtime_checkable
class Display(Protocol):
    """A display that can be styled and plotted."""

    def plot(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        """Plot the display."""

    def set_style(self, **kwargs: Any) -> None:
        """Set the style of the display."""
