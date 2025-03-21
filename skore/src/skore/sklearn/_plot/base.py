from typing import Any, Optional, Protocol, runtime_checkable

from matplotlib.axes import Axes


@runtime_checkable
class Display(Protocol):
    """Protocol for the public API of display objects."""

    def plot(self, ax: Optional[Axes] = None, **kwargs: Any) -> None:
        """Plot the display."""
        ...

    def set_style(self, **kwargs: Any):
        """Set the style of the display."""
        ...
