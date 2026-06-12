"""The picker backing interactive ``skore skills update`` / ``remove``."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Label, SelectionList
from textual.widgets.selection_list import Selection

_INTRO = (
    "Select the installed skills to manage.\n"
    "[reverse] ↑/↓ [/] move  [reverse] Space [/] (de)select  "
    "[reverse] Enter [/] confirm"
)


class InstalledSkillsPicker(App[list[str] | None]):
    """Pick installed skill ids from a single selection list."""

    CSS = """
    Screen {
        align: center middle;
    }
    #picker {
        width: 90%;
        height: 90%;
    }
    .picker-intro {
        margin: 1 1;
        color: $text-muted;
    }
    #sel-installed {
        height: 1fr;
        border: round $accent;
        padding: 0 1;
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("enter", "confirm", "Confirm", priority=True),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, skill_ids: list[str], *, title: str) -> None:
        super().__init__()
        self._skill_ids = skill_ids
        self._title = title
        self.result: list[str] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="picker"):
            yield Label(self._title, classes="picker-intro")
            yield Label(_INTRO, classes="picker-intro")
            skill_list = SelectionList[str](
                *(Selection(skill_id, skill_id) for skill_id in self._skill_ids),
                id="sel-installed",
            )
            skill_list.border_title = "Installed skills"
            yield skill_list
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#sel-installed", SelectionList).focus()

    def action_confirm(self) -> None:
        selected = list(self.query_one("#sel-installed", SelectionList).selected)
        if not selected:
            self.notify(
                "Select at least one skill.",
                severity="warning",
            )
            return
        self.result = selected
        self.exit()

    def action_cancel(self) -> None:
        self.result = None
        self.exit()
