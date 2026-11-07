"""A radio set that selects the highlighted option on arrow navigation."""

from __future__ import annotations

from textual.widgets import RadioButton, RadioSet


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
