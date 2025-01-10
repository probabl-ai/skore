from typing import Any, Literal, Optional

import numpy as np
from rich.panel import Panel
from rich.tree import Tree

class _HelpMixin:
    def _get_methods_for_help(self) -> list[tuple[str, Any]]: ...
    def _sort_methods_for_help(
        self, methods: list[tuple[str, Any]]
    ) -> list[tuple[str, Any]]: ...
    def _format_method_name(self, name: str) -> str: ...
    def _get_method_description(self, method: Any) -> str: ...
    def _create_help_panel(self) -> Panel: ...
    def _get_help_panel_title(self) -> str: ...
    def _create_help_tree(self) -> Tree: ...
    def help(self) -> None: ...
    def __repr__(self) -> str: ...

class _BaseAccessor(_HelpMixin):
    _parent: Any
    _icon: str

    def __init__(self, parent: Any, icon: str) -> None: ...
    def _get_help_panel_title(self) -> str: ...
    def _create_help_tree(self) -> Tree: ...
    def _get_X_y_and_data_source_hash(
        self,
        *,
        data_source: Literal["test", "train", "X_y"],
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]: ...
