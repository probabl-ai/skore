"""Modal help overlay for interactive CLI screens."""

from __future__ import annotations

from typing import Protocol, cast

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, Label, Static

HELP_BINDING = Binding("question_mark", "show_help", "Help", priority=True)


class _HelpApp(Protocol):
    def action_show_help(self) -> None: ...


class HelpInput(Input):
    """Input that opens the app help screen instead of inserting ``?``."""

    async def _on_key(self, event: Key) -> None:
        if event.key == "question_mark":
            cast(_HelpApp, self.app).action_show_help()
            event.prevent_default()
            event.stop()
            return
        await super()._on_key(event)


class HelpScreen(ModalScreen[None]):
    """A dismissible help panel opened with ``?``."""

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    #help-panel {
        width: 80%;
        max-width: 90;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    .help-title {
        text-style: bold;
        margin-bottom: 1;
    }
    .help-body {
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close"),
    ]

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="help-panel"):
            yield Label(self._title, classes="help-title")
            yield Static(self._body, classes="help-body")
        yield Footer()

    async def action_dismiss(self, result: None = None) -> None:
        self.dismiss(result)
