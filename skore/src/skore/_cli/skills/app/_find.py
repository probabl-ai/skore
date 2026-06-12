"""The Textual finder backing ``skore skills find``."""

from __future__ import annotations

from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, SelectionList

from skore._cli.skills.app._widgets import SkillSelection

_FIND_INTRO = (
    "Browse the catalog and pick the workflows and individual skills you want "
    "to preview.\n"
    "[reverse] ↑/↓ [/] move  [reverse] Space [/] (de)select  "
    "[reverse] Tab [/] switch lists  [reverse] Enter [/] show details"
)


class ProbablSkillsFinder(App[None]):
    """A single-step selector to preview catalog entries.

    Reuses the installer's skill picker; confirming returns the chosen ids so
    the caller can render their id and description as a table.
    """

    CSS = """
    Screen {
        align: center middle;
    }
    #finder {
        width: 90%;
        height: 90%;
    }
    """

    BINDINGS = [
        Binding("enter", "confirm", "Confirm", priority=True),
        Binding("tab", "focus_next", "Switch list", show=True),
        Binding("shift+tab", "focus_previous", "Switch list", show=True),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, catalog: dict[str, Any]) -> None:
        super().__init__()
        self._catalog = catalog
        self.result: list[str] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield SkillSelection(self._catalog, intro=_FIND_INTRO, id="finder")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#sel-workflows", SelectionList).focus()

    def action_confirm(self) -> None:
        selected = self.query_one(SkillSelection).selected_ids()
        if not selected:
            self.notify(
                "Select at least one workflow or skill.",
                severity="warning",
            )
            return
        self.result = selected
        self.exit()

    def action_cancel(self) -> None:
        self.result = None
        self.exit()
