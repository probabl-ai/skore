"""Shared Textual widgets for the skills installer and finder apps."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, RadioButton, RadioSet, SelectionList
from textual.widgets.selection_list import Selection


class AutoRadioSet(RadioSet):
    """A radio set that selects the highlighted option on arrow navigation."""

    def _sync_pressed_from_selected(self) -> None:
        if self._selected is not None:
            button = self._nodes[self._selected]
            if isinstance(button, RadioButton) and not button.value:
                button.value = True

    def select_index(self, index: int) -> None:
        """Set highlight and pressed selection to the button at ``index``."""
        self.focus()
        self._selected = index
        self._sync_pressed_from_selected()

    def action_next_button(self) -> None:
        super().action_next_button()
        self._sync_pressed_from_selected()

    def action_previous_button(self) -> None:
        super().action_previous_button()
        self._sync_pressed_from_selected()

    def watch__selected(self) -> None:
        super().watch__selected()
        self._sync_pressed_from_selected()


class SkillSelection(Vertical):
    """Two synced selection lists for workflows and individual skills.

    Selecting a workflow also selects its bundled skills; deselecting it
    removes the ones that no other selected workflow still requires.
    """

    DEFAULT_CSS = """
    SkillSelection {
        height: 1fr;
    }
    SkillSelection > SelectionList {
        height: 1fr;
        border: round $surface-lighten-2;
        padding: 0 1;
        margin: 0 1;
    }
    SkillSelection > SelectionList:focus {
        border: thick $accent;
    }
    SkillSelection > .skills-intro {
        margin: 1 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        catalog: dict[str, Any],
        *,
        intro: str | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._catalog = catalog
        self._intro = intro
        self._workflows = {
            workflow["id"]: workflow for workflow in catalog.get("workflows", [])
        }
        self._skill_selection: dict[str, Selection[str]] = {}
        self._selected_workflows: set[str] = set()

    def compose(self) -> ComposeResult:
        if self._intro is not None:
            yield Label(self._intro, classes="skills-intro")

        workflow_selections = [
            Selection(f"{entry['id']}  —  {entry.get('summary', '')}", entry["id"])
            for entry in self._catalog.get("workflows", [])
        ]
        workflow_list = SelectionList[str](*workflow_selections, id="sel-workflows")
        workflow_list.border_title = "Workflows"
        yield workflow_list

        skill_selections = []
        for entry in self._catalog.get("skills", []):
            selection = Selection(
                f"{entry['id']}  —  {entry.get('summary', '')}", entry["id"]
            )
            self._skill_selection[entry["id"]] = selection
            skill_selections.append(selection)
        skill_list = SelectionList[str](*skill_selections, id="sel-skills")
        skill_list.border_title = "Individual skills"
        yield skill_list

    def on_selection_list_selected_changed(
        self, event: SelectionList.SelectedChanged
    ) -> None:
        if event.selection_list.id == "sel-workflows":
            self._sync_workflow_skills()

    def _sync_workflow_skills(self) -> None:
        skill_list = self.query_one("#sel-skills", SelectionList)
        current = set(self.query_one("#sel-workflows", SelectionList).selected)
        added = current - self._selected_workflows
        removed = self._selected_workflows - current
        self._selected_workflows = current
        with self.prevent(SelectionList.SelectedChanged):
            for workflow_id in added:
                for skill_id in self._workflows[workflow_id]["includes"]:
                    selection = self._skill_selection.get(skill_id)
                    if selection is not None:
                        skill_list.select(selection)
            for workflow_id in removed:
                for skill_id in self._workflows[workflow_id]["includes"]:
                    still_required = any(
                        skill_id in self._workflows[other]["includes"]
                        for other in current
                    )
                    selection = self._skill_selection.get(skill_id)
                    if not still_required and selection is not None:
                        skill_list.deselect(selection)

    def selected_ids(self) -> list[str]:
        workflows = self.query_one("#sel-workflows", SelectionList).selected
        skills = self.query_one("#sel-skills", SelectionList).selected
        return list(workflows) + list(skills)
