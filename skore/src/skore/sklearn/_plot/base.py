from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Display(Protocol):
    """Protocol specifying the common API for all `skore` displays."""

    def plot(self, **kwargs: Any) -> None:
        """Display a matplotlib figure containing the information of the display."""

    def set_style(self, **kwargs: Any) -> None:
        """Set the style of the display."""
