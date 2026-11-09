"""Shared SKORE banner for Textual applications."""

from textual.widgets import Static

from skore._cli.style import SKORE_BANNER


class SkoreBanner(Static):
    """Display the SKORE wordmark."""

    DEFAULT_CSS = """
    SkoreBanner {
        width: 100%;
        height: 5;
        content-align: center middle;
        color: $accent;
        text-style: bold;
    }
    """

    def __init__(self) -> None:
        super().__init__(SKORE_BANNER.strip("\n"), markup=False)
