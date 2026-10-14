"""The tabbed Textual installer backing ``skore skills install``."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import (
    Footer,
    Header,
    Label,
    RadioButton,
    SelectionList,
    TabbedContent,
    TabPane,
)

from skore._cli.agents import DEFAULT_AGENT, SKILL_AGENT_NAMES
from skore._cli.app.banner import SkoreBanner
from skore._cli.app.help import HELP_BINDING, HelpInput, HelpScreen
from skore._cli.skills.app.widgets import AutoRadioSet, SkillSelection
from skore._cli.skills.catalog import GITHUB_REPO, normalize_github_repo
from skore._cli.skills.release import ReleaseError, cleanup_release, load_release

_SOURCE_INTRO = (
    "GitHub repository that publishes the skills catalog (owner/name).\n"
    "The default is probabl-ai/skore-skills. Change it to install from a fork "
    "or another catalog.\n"
    "[reverse] Enter [/] load catalog  [reverse] ? [/] help"
)

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
    "[reverse] ↑/↓ [/] choose  [reverse] Enter [/] confirm  [reverse] ? [/] help"
)

_INSTALL_HELP = """\
Install skills in four steps:

1. Choose the GitHub owner/name that publishes the catalog
2. Pick workflows and/or individual skills
3. Choose the target agent directory
4. Choose project-local or global scope

Keys:
  ↑/↓ Space move and (de)select
  Tab       switch fields or wizard tabs
  Enter     confirm step
  Esc       cancel
  ?         show this help
"""


class ProbablSkillsInstaller(App[None]):
    """A tabbed wizard to pick a GitHub source, skills, agents and scope."""

    CSS = """
    Screen {
        align: center middle;
    }
    #installer {
        width: 90%;
        height: 90%;
    }
    #wizard {
        width: 100%;
        height: 1fr;
    }
    .step-intro {
        margin: 1 1;
        color: $text-muted;
    }
    #repo {
        margin: 0 1 1 1;
        width: 100%;
    }
    #skills-host {
        height: 1fr;
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
        HELP_BINDING,
    ]

    def __init__(
        self,
        *,
        agent: tuple[str, ...],
        default_global: bool,
        default_repo: str = GITHUB_REPO,
    ) -> None:
        super().__init__()
        self._agent_names_cli = list(agent)
        self._ask_agent = not agent
        self._default_global = default_global
        self._default_repo = default_repo
        self._repo: str | None = None
        self._tag: str | None = None
        self._root: Path | None = None
        self._catalog: dict[str, Any] | None = None
        self.result: tuple[list[str], list[str], bool, str] | None = None

    @property
    def tag(self) -> str | None:
        """Return the fetched release tag, if any."""
        return self._tag

    @property
    def root(self) -> Path | None:
        """Return the extracted release root, if any."""
        return self._root

    @property
    def catalog(self) -> dict[str, Any] | None:
        """Return the fetched catalog, if any."""
        return self._catalog

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="installer"):
            yield SkoreBanner()
            with TabbedContent(id="wizard"):
                with TabPane("1 · Source", id="step-source"):
                    yield Label(_SOURCE_INTRO, classes="step-intro")
                    yield HelpInput(
                        value=self._default_repo,
                        placeholder="owner/name",
                        id="repo",
                    )
                with TabPane("2 · Skills", id="step-skills"):
                    yield Vertical(id="skills-host")
                if self._ask_agent:
                    with TabPane("3 · Agents", id="step-agents"):
                        yield Label(_AGENTS_INTRO, classes="step-intro")
                        with AutoRadioSet(id="agents"):
                            for name in SKILL_AGENT_NAMES:
                                recommended = name == DEFAULT_AGENT
                                label = (
                                    f"{name}  (recommended — open standard)"
                                    if recommended
                                    else name
                                )
                                yield RadioButton(label, value=recommended)
                scope_label = "4 · Scope" if self._ask_agent else "3 · Scope"
                with TabPane(scope_label, id="step-scope"):
                    yield Label(_SCOPE_INTRO, classes="step-intro")
                    with AutoRadioSet(id="scope"):
                        yield RadioButton(
                            "Project (local)", value=not self._default_global
                        )
                        yield RadioButton("User (global)", value=self._default_global)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#repo", HelpInput).focus()

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen("Install skills", _INSTALL_HELP))

    def _selected_ids(self) -> list[str]:
        try:
            return self.query_one(SkillSelection).selected_ids()
        except NoMatches:
            return []

    def _selected_agents(self) -> list[str]:
        index = self.query_one("#agents", AutoRadioSet).pressed_index
        return [SKILL_AGENT_NAMES[index]] if index >= 0 else []

    def _ensure_agents_selected(self) -> None:
        radio = self.query_one("#agents", AutoRadioSet)
        if radio.pressed_index < 0:
            radio.select_index(SKILL_AGENT_NAMES.index(DEFAULT_AGENT))

    def _ensure_scope_selected(self) -> None:
        radio = self.query_one("#scope", AutoRadioSet)
        if radio.pressed_index < 0:
            radio.select_index(1 if self._default_global else 0)

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        if event.tabbed_content.id != "wizard":
            return
        wizard = self.query_one("#wizard", TabbedContent)
        if wizard.active != "step-source" and self._catalog is None:
            wizard.active = "step-source"
            self.notify("Confirm a GitHub source first.", severity="warning")
            return
        self._focus_active_step()

    def _focus_active_step(self) -> None:
        active = self.query_one("#wizard", TabbedContent).active
        if active == "step-source":
            self.query_one("#repo", HelpInput).focus()
        elif active == "step-skills":
            with suppress(NoMatches):
                self.query_one("#sel-workflows", SelectionList).focus()
        elif active == "step-agents":
            radio = self.query_one("#agents", AutoRadioSet)
            radio.select_index(SKILL_AGENT_NAMES.index(DEFAULT_AGENT))
        elif active == "step-scope":
            radio = self.query_one("#scope", AutoRadioSet)
            radio.select_index(1 if self._default_global else 0)

    def _cleanup_release(self) -> None:
        cleanup_release(self._root)
        self._root = None
        self._tag = None
        self._catalog = None
        self._repo = None

    async def _mount_skills(self, catalog: dict[str, Any]) -> None:
        host = self.query_one("#skills-host", Vertical)
        await host.remove_children()
        await host.mount(SkillSelection(catalog, intro=_SKILLS_INTRO))

    async def _load_source(self) -> bool:
        raw = self.query_one("#repo", HelpInput).value
        try:
            repo = normalize_github_repo(raw)
        except ValueError:
            self.notify(
                "GitHub repository must be owner/name.",
                severity="warning",
            )
            return False

        if self._repo == repo and self._catalog is not None:
            return True

        self.notify(f"Fetching latest skills release from {repo}...")
        try:
            tag, root, catalog = await asyncio.to_thread(load_release, repo)
        except ReleaseError as error:
            self.notify(str(error), severity="error")
            return False

        self._cleanup_release()
        self._repo = repo
        self._tag = tag
        self._root = root
        self._catalog = catalog
        await self._mount_skills(catalog)
        return True

    def _finish(self) -> None:
        self._ensure_scope_selected()
        agent_names = (
            self._selected_agents() if self._ask_agent else self._agent_names_cli
        )
        global_ = self.query_one("#scope", AutoRadioSet).pressed_index == 1
        assert self._repo is not None
        self.result = (self._selected_ids(), agent_names, global_, self._repo)
        self.exit()

    async def action_confirm(self) -> None:
        wizard = self.query_one("#wizard", TabbedContent)
        active = wizard.active
        if active == "step-source":
            if not await self._load_source():
                return
            wizard.active = "step-skills"
            self._focus_active_step()
            self.call_after_refresh(self._focus_active_step)
        elif active == "step-skills":
            if not self._selected_ids():
                self.notify(
                    "Select at least one workflow or skill.",
                    severity="warning",
                )
                return
            wizard.active = "step-agents" if self._ask_agent else "step-scope"
            self._focus_active_step()
            self.call_after_refresh(self._focus_active_step)
        elif active == "step-agents":
            self._ensure_agents_selected()
            if not self._selected_agents():
                self.notify("Select an agent.", severity="warning")
                return
            wizard.active = "step-scope"
            self._focus_active_step()
            self.call_after_refresh(self._focus_active_step)
        elif active == "step-scope":
            self._finish()

    def action_cancel(self) -> None:
        self._cleanup_release()
        self.result = None
        self.exit()
