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
    RadioSet,
    SelectionList,
    TabbedContent,
    TabPane,
)

from skore._cli.skills._agents import AGENT_NAMES, DEFAULT_AGENT
from skore._cli.skills.app._widgets import SkillSelection

_SKILLS_INTRO = (
    "[b]Workflows[/b] bundle several related skills; selecting a workflow also "
    "selects its [b]individual skills[/b] below.\n"
    "Use [b]↑/↓[/b] to move, [b]Space[/b] to (de)select, [b]Tab[/b] to switch "
    "between the two lists, and [b]Enter[/b] to confirm."
)

_AGENTS_INTRO = (
    "Choose the [b]agent[/b] to install for.\n"
    "[b]agents[/b] targets the [b].agents/[/b] directory, the cross-client open "
    "standard, and is recommended.\n"
    "Use [b]↑/↓[/b] and [b]Space[/b] to choose one, then [b]Enter[/b] to confirm."
)

_SCOPE_INTRO = (
    "[b]Project (local)[/b] installs into the current repository only.\n"
    "[b]User (global)[/b] installs into your home directory so every project "
    "can use the skills.\n"
    "Use [b]↑/↓[/b] and [b]Space[/b] to choose one, then [b]Enter[/b] to confirm."
)


class ProbablSkillsInstaller(App[None]):
    """A tabbed wizard to pick skills, target agents and the install scope.

    Navigation mirrors a question flow: arrows move within a list, ``Space``
    toggles an entry and ``Enter`` always confirms the current step.
    """

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
    RadioSet {
        margin: 1 1;
        width: 100%;
    }
    """

    BINDINGS = [
        Binding("enter", "confirm", "Confirm", priority=True),
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
                    with RadioSet(id="agents"):
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
                with RadioSet(id="scope"):
                    yield RadioButton("Project (local)", value=not self._default_global)
                    yield RadioButton("User (global)", value=self._default_global)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#sel-workflows", SelectionList).focus()

    def _selected_ids(self) -> list[str]:
        return self.query_one(SkillSelection).selected_ids()

    def _selected_agents(self) -> list[str]:
        index = self.query_one("#agents", RadioSet).pressed_index
        return [AGENT_NAMES[index]] if index >= 0 else []

    def _focus_active_step(self) -> None:
        active = self.query_one("#wizard", TabbedContent).active
        if active == "step-agents":
            self.query_one("#agents", RadioSet).focus()
        elif active == "step-scope":
            self.query_one("#scope", RadioSet).focus()

    def _finish(self) -> None:
        agent_names = (
            self._selected_agents() if self._ask_agent else self._agent_names_cli
        )
        global_ = self.query_one("#scope", RadioSet).pressed_index == 1
        self.result = (self._selected_ids(), agent_names, global_)
        self.exit()

    def action_confirm(self) -> None:
        wizard = self.query_one("#wizard", TabbedContent)
        active = wizard.active
        if active == "step-skills":
            if not self._selected_ids():
                self.notify(
                    "Select at least one workflow or skill (press Space).",
                    severity="warning",
                )
                return
            wizard.active = "step-agents" if self._ask_agent else "step-scope"
            self.call_after_refresh(self._focus_active_step)
        elif active == "step-agents":
            if not self._selected_agents():
                self.notify(
                    "Select at least one agent (press Space).", severity="warning"
                )
                return
            wizard.active = "step-scope"
            self.call_after_refresh(self._focus_active_step)
        elif active == "step-scope":
            self._finish()

    def action_cancel(self) -> None:
        self.result = None
        self.exit()
