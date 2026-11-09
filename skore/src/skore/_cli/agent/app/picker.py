"""The single-select picker backing interactive ``skore agent``."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Label, RadioButton

from skore._cli.app.banner import SkoreBanner
from skore._cli.app.help import HELP_BINDING, HelpScreen
from skore._cli.app.widgets import AutoRadioSet

_HARNESS_INTRO = (
    "Choose the agent harness to launch.\n"
    "[reverse] ↑/↓ [/] choose  [reverse] Enter [/] confirm  [reverse] ? [/] help"
)

_HARNESS_HELP = """\
Pick the local coding agent to configure and launch.

Detected harnesses are listed under Detected.
Harnesses that are not installed are listed under Other.

Supported harnesses:
  • Bob Shell    — writes .bob/mcp.json
  • Bob IDE      — writes .bob/mcp.json
  • Claude CLI   — writes .claude/settings.local.json
  • Claude UI    — writes .claude/settings.local.json
  • Claude Plugin — writes .claude/settings.local.json
  • Cursor IDE   — writes .cursor/mcp.json
  • Cursor CLI   — writes .cursor/mcp.json
  • OpenCode     — writes opencode.json
  • Pi           — writes .pi/agent/models.json
  • Copilot in VSCode — writes .vscode/chatLanguageModels.json
  • Copilot CLI  — launches copilot with Hub provider env
  • Codex CLI    — writes .codex/skore-provider.toml (+ ~/.codex/config.toml)

Skore stores your hub credentials in .skore and selects the
skore-agent model when the harness starts.

Keys:
  ↑/↓     move selection
  Enter   confirm
  Esc     cancel
  ?       show this help
"""

_WORKSPACE_INTRO = (
    "Choose the Skore Hub workspace to attach the agent to.\n"
    "The agent uses this workspace's LLM provider configuration.\n"
    "[reverse] ↑/↓ [/] choose  [reverse] Enter [/] confirm  [reverse] ? [/] help"
)

_WORKSPACE_HELP = """\
Pick the hub workspace this project should use.

Skore creates a workspace-scoped API key and saves it in
.skore together with the workspace id.

Keys:
  ↑/↓     move selection
  Enter   confirm
  Esc     cancel
  ?       show this help
"""


class HarnessPicker(App[str | None]):
    """Pick a single harness name from a radio set."""

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
    AutoRadioSet {
        margin: 1 1;
        width: 100%;
    }
    AutoRadioSet > .harness-group {
        color: $text-muted;
        text-style: bold;
        height: auto;
        margin: 1 0 0 0;
    }
    AutoRadioSet > .harness-group:first-child {
        margin-top: 0;
    }
    """

    BINDINGS = [
        Binding("enter", "confirm", "Confirm", priority=True),
        Binding("escape", "cancel", "Cancel"),
        HELP_BINDING,
    ]

    def __init__(
        self,
        harnesses: list[tuple[str, str, bool]],
        *,
        preselect: int = 0,
    ) -> None:
        super().__init__()
        self._harnesses = harnesses
        self._preselect = preselect
        self.result: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="picker"):
            yield SkoreBanner()
            yield Label(_HARNESS_INTRO, classes="picker-intro")
            with AutoRadioSet(id="harnesses"):
                for title, entries in self._sections():
                    yield Label(f"{title}:", classes="harness-group", disabled=True)
                    for name, label in entries:
                        yield RadioButton(label, name=name)
        yield Footer()

    def _sections(self) -> list[tuple[str, list[tuple[str, str]]]]:
        detected = [(name, label) for name, label, found in self._harnesses if found]
        other = [(name, label) for name, label, found in self._harnesses if not found]
        sections = []
        if detected:
            sections.append(("Detected", detected))
        if other:
            sections.append(("Other", other))
        return sections

    def on_mount(self) -> None:
        radio = self.query_one("#harnesses", AutoRadioSet)
        if not self._harnesses:
            return
        target = self._harnesses[self._preselect][0]
        for index, child in enumerate(radio.children):
            if isinstance(child, RadioButton) and child.name == target:
                radio.select_index(index)
                return

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen("Choose a harness", _HARNESS_HELP))

    def action_confirm(self) -> None:
        pressed = self.query_one("#harnesses", AutoRadioSet).pressed_button
        if pressed is None or not pressed.name:
            self.notify("Select a harness.", severity="warning")
            return
        self.result = pressed.name
        self.exit()

    def action_cancel(self) -> None:
        self.result = None
        self.exit()


class WorkspacePicker(App[str | None]):
    """Pick a single workspace public id from a radio set."""

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
    AutoRadioSet {
        margin: 1 1;
        width: 100%;
    }
    """

    BINDINGS = [
        Binding("enter", "confirm", "Confirm", priority=True),
        Binding("escape", "cancel", "Cancel"),
        HELP_BINDING,
    ]

    def __init__(
        self,
        workspaces: list[tuple[str, str]],
        *,
        preselect: int = 0,
    ) -> None:
        super().__init__()
        self._workspaces = workspaces
        self._preselect = preselect
        self.result: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="picker"):
            yield SkoreBanner()
            yield Label(_WORKSPACE_INTRO, classes="picker-intro")
            with AutoRadioSet(id="workspaces"):
                for index, (public_id, name) in enumerate(self._workspaces):
                    text = f"{name}  ({public_id})" if name != public_id else public_id
                    yield RadioButton(text, value=index == self._preselect)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#workspaces", AutoRadioSet).select_index(self._preselect)

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen("Choose a workspace", _WORKSPACE_HELP))

    def action_confirm(self) -> None:
        index = self.query_one("#workspaces", AutoRadioSet).pressed_index
        if index < 0:
            self.notify("Select a workspace.", severity="warning")
            return
        self.result = self._workspaces[index][0]
        self.exit()

    def action_cancel(self) -> None:
        self.result = None
        self.exit()


_IDE_INTRO = (
    "Open Claude Plugin in:\n"
    "[reverse] ↑/↓ [/] choose  [reverse] Enter [/] confirm  [reverse] ? [/] help"
)

_IDE_HELP = """\
Pick the IDE that should open the Claude Code panel.

Only IDEs that have the Claude plugin installed are listed.

Keys:
  ↑/↓     move selection
  Enter   confirm
  Esc     cancel
  ?       show this help
"""


class IdePicker(App[str | None]):
    """Pick the IDE that should open the Claude plugin."""

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
    AutoRadioSet {
        margin: 1 1;
        width: 100%;
    }
    """

    BINDINGS = [
        Binding("enter", "confirm", "Confirm", priority=True),
        Binding("escape", "cancel", "Cancel"),
        HELP_BINDING,
    ]

    def __init__(
        self,
        ides: list[tuple[str, str]],
        *,
        preselect: int = 0,
    ) -> None:
        super().__init__()
        self._ides = ides
        self._preselect = preselect
        self.result: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="picker"):
            yield SkoreBanner()
            yield Label(_IDE_INTRO, classes="picker-intro")
            with AutoRadioSet(id="ides"):
                for index, (_, label) in enumerate(self._ides):
                    yield RadioButton(label, value=index == self._preselect)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#ides", AutoRadioSet).select_index(self._preselect)

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen("Open Claude Plugin", _IDE_HELP))

    def action_confirm(self) -> None:
        index = self.query_one("#ides", AutoRadioSet).pressed_index
        if index < 0:
            self.notify("Select an IDE.", severity="warning")
            return
        self.result = self._ides[index][0]
        self.exit()

    def action_cancel(self) -> None:
        self.result = None
        self.exit()
