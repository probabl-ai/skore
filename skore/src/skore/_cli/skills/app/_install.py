"""The tabbed Textual installer backing ``skore skills install``."""

from __future__ import annotations

from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import (
    Footer,
    Header,
    Label,
    RadioButton,
    SelectionList,
    TabbedContent,
    TabPane,
)

from skore._cli.skills._agents import AGENT_NAMES, DEFAULT_AGENT
from skore._cli.skills.app._widgets import AutoRadioSet, SkillSelection

_SKILLS_INTRO = (
    "Workflows bundle several related skills; selecting a workflow also selects "
    "its individual skills below.\n"
    "[reverse] ↑/↓ [/] move  [reverse] Space [/] (de)select  "
    "[reverse] Tab [/] switch lists  [reverse] Enter [/] confirm"
)

_AGENTS_INTRO = (
    "Choose the agent to install for.\n"
    "agents targets the .agents/ directory, the cross-client open standard, "
    "and is recommended.\n"
    "[reverse] ↑/↓ [/] choose  [reverse] Enter [/] confirm"
)

_SCOPE_INTRO = (
    "Project (local) installs into the current repository only.\n"
    "User (global) installs into your home directory so every project can use "
    "the skills.\n"
    "[reverse] ↑/↓ [/] choose  [reverse] Enter [/] confirm"
)


class ProbablSkillsInstaller(App[None]):
    """A tabbed wizard to pick skills, target agents and the install scope."""

    CSS = """
    Screen {
        align: center middle;
    }
    #wizard {
        width: 90%;
        height: 90%;
    }
    .step-intro {
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
        Binding("tab", "focus_next", "Switch list", show=True),
        Binding("shift+tab", "focus_previous", "Switch list", show=True),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        catalog: dict[str, Any],
        *,
        agent: tuple[str, ...],
        default_global: bool,
    ) -> None:
        super().__init__()
        self._catalog = catalog
        self._agent_names_cli = list(agent)
        self._ask_agent = not agent
        self._default_global = default_global
        self.result: tuple[list[str], list[str], bool] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(id="wizard"):
            with TabPane("1 · Skills", id="step-skills"):
                yield SkillSelection(self._catalog, intro=_SKILLS_INTRO)
            if self._ask_agent:
                with TabPane("2 · Agents", id="step-agents"):
                    yield Label(_AGENTS_INTRO, classes="step-intro")
                    with AutoRadioSet(id="agents"):
                        for name in AGENT_NAMES:
                            recommended = name == DEFAULT_AGENT
                            label = (
                                f"{name}  (recommended — open standard)"
                                if recommended
                                else name
                            )
                            yield RadioButton(label, value=recommended)
            with TabPane("3 · Scope", id="step-scope"):
                yield Label(_SCOPE_INTRO, classes="step-intro")
                with AutoRadioSet(id="scope"):
                    yield RadioButton("Project (local)", value=not self._default_global)
                    yield RadioButton("User (global)", value=self._default_global)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#sel-workflows", SelectionList).focus()

    def _selected_ids(self) -> list[str]:
        return self.query_one(SkillSelection).selected_ids()

    def _selected_agents(self) -> list[str]:
        index = self.query_one("#agents", AutoRadioSet).pressed_index
        return [AGENT_NAMES[index]] if index >= 0 else []

    def _focus_active_step(self) -> None:
        active = self.query_one("#wizard", TabbedContent).active
        if active == "step-agents":
            radio = self.query_one("#agents", AutoRadioSet)
            radio.select_index(AGENT_NAMES.index(DEFAULT_AGENT))
        elif active == "step-scope":
            radio = self.query_one("#scope", AutoRadioSet)
            radio.select_index(1 if self._default_global else 0)

    def _finish(self) -> None:
        agent_names = (
            self._selected_agents() if self._ask_agent else self._agent_names_cli
        )
        global_ = self.query_one("#scope", AutoRadioSet).pressed_index == 1
        self.result = (self._selected_ids(), agent_names, global_)
        self.exit()

    def action_confirm(self) -> None:
        wizard = self.query_one("#wizard", TabbedContent)
        active = wizard.active
        if active == "step-skills":
            if not self._selected_ids():
                self.notify(
                    "Select at least one workflow or skill.",
                    severity="warning",
                )
                return
            wizard.active = "step-agents" if self._ask_agent else "step-scope"
            self.call_after_refresh(self._focus_active_step)
        elif active == "step-agents":
            if not self._selected_agents():
                self.notify("Select an agent.", severity="warning")
                return
            wizard.active = "step-scope"
            self.call_after_refresh(self._focus_active_step)
        elif active == "step-scope":
            self._finish()

    def action_cancel(self) -> None:
        self.result = None
        self.exit()
