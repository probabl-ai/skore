"""The command menu backing ``skore skills`` with no subcommand."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Label, OptionList
from textual.widgets.option_list import Option

_COMMANDS: tuple[tuple[str, str], ...] = (
    ("find", "Search available skills and workflows in the latest release"),
    ("list", "List installed skills"),
    ("install", "Install skills or workflows from the latest release"),
    ("update", "Update installed skills to the latest release"),
    ("remove", "Remove installed skills"),
)


class SkillsMenu(App[str | None]):
    """Pick a ``skore skills`` subcommand from a visible menu."""

    CSS = """
    Screen {
        align: center middle;
    }
    #menu-panel {
        width: 70;
        height: auto;
        max-height: 90%;
        border: round $accent;
        padding: 1 2;
    }
    #menu-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    #menu-hint {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }
    OptionList {
        height: auto;
        max-height: 1fr;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.result: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="menu-panel"):
            yield Label("Agent Skills", id="menu-title")
            yield Label(
                "Choose a command to run interactively.",
                id="menu-hint",
            )
            yield OptionList(
                *(
                    Option(f"[b cyan]{name}[/]  {description}", id=name)
                    for name, description in _COMMANDS
                ),
                id="commands",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#commands", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.result = str(event.option_id)
        self.exit()

    def action_cancel(self) -> None:
        self.result = None
        self.exit()
