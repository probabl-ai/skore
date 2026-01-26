"""Rich-based help rendering mixins."""

from abc import ABC, abstractmethod
from dataclasses import asdict
from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from skore._utils.repr.data import (
    _AccessorHelpDataMixin,
    _DisplayHelpDataMixin,
    _ReportHelpDataMixin,
)


class _BaseRichHelpMixin(ABC):
    """Base mixin for Rich-based help rendering."""

    _ACCESSOR_CONFIG: dict[str, dict[str, str]]

    @abstractmethod
    def _create_help_tree(self) -> Tree:
        """Create the help tree for Rich rendering."""

    @abstractmethod
    def _create_help_panel(self) -> Panel:
        """Create the Rich panel wrapping the help tree."""


class _RichReportHelpMixin(_ReportHelpDataMixin, _BaseRichHelpMixin):
    """Mixin for Rich-based help rendering for reports."""

    def _create_help_tree(self) -> Tree:
        """Create the help tree for Rich rendering (reports only)."""
        data = asdict(self._build_help_data())
        data["is_report"] = True

        tree = Tree(data["root_node"])

        for accessor_data in data.get("accessors", []):
            branch = tree.add(f"[bold cyan].{accessor_data['name']}[/bold cyan]")

            for method in accessor_data["methods"]:
                displayed_name = f"{method['name']}(...)"
                description = method["description"]
                branch.add(f".{displayed_name.ljust(25)} - {description}")

        base_methods = data.get("base_methods", [])
        if base_methods:
            methods_branch = tree.add("[bold cyan]Methods[/bold cyan]")
            for method in base_methods:
                displayed_name = f"{method['name']}(...)"
                description = method["description"]
                methods_branch.add(f".{displayed_name}".ljust(26) + f" - {description}")

        attributes = data.get("attributes") or []
        if attributes:
            attr_branch = tree.add("[bold cyan]Attributes[/bold cyan]")
            for attr in attributes:
                name = attr["name"]
                description = attr["description"]
                attr_branch.add(f".{name.ljust(25)} - {description}")

        return tree

    def _create_help_panel(self) -> Panel:
        """Create the Rich panel wrapping the report help tree."""
        content = self._create_help_tree()

        title_plain = self._get_help_title()
        title_rich = (
            f"[bold cyan]{title_plain}[/bold cyan]" if title_plain else title_plain
        )

        return Panel(
            content,
            title=title_rich,
            expand=False,
            border_style="orange1",
        )


class _RichAccessorHelpMixin(_AccessorHelpDataMixin, _BaseRichHelpMixin):
    """Mixin for Rich-based help rendering for accessors."""

    def _create_help_tree(self) -> Tree:
        """Create the help tree for Rich rendering for accessors."""
        data = asdict(self._build_help_data())
        data["is_report"] = False

        tree = Tree(data["root_node"])

        # Add accessor branch
        accessor_name = data.get("accessor_name", "")
        branch = tree.add(f"[bold cyan].{accessor_name}[/bold cyan]")

        # Add methods under the accessor branch
        for method in data.get("methods", []):
            displayed_name = f"{method['name']}(...)"
            description = method["description"]
            branch.add(f".{displayed_name}".ljust(26) + f" - {description}")

        return tree

    def _create_help_panel(self) -> Panel:
        """Create the Rich panel wrapping the accessor help tree."""
        content = self._create_help_tree()

        title_plain = self._get_help_title()
        title_rich = (
            f"[bold cyan]{title_plain}[/bold cyan]" if title_plain else title_plain
        )

        return Panel(
            content,
            title=title_rich,
            expand=False,
            border_style="orange1",
        )


class _RichHelpDisplayMixin(_DisplayHelpDataMixin, _BaseRichHelpMixin):
    """Mixin for Rich-based help rendering for displays."""

    def _create_help_tree(self) -> Tree:
        """Create a rich Tree with attributes and methods."""
        data = asdict(self._build_help_data())

        tree = Tree(f"[bold cyan]{data['root_node']}[/bold cyan]")

        methods = data.get("methods") or []
        if methods:
            method_branch = tree.add("[bold cyan]Methods[/bold cyan]")
            for method in methods:
                name = f".{method['name']}(...)"
                description = method["description"]
                method_branch.add(f"{name.ljust(26)} - {description}")

        attributes = data.get("attributes") or []
        if attributes:
            attr_branch = tree.add("[bold cyan]Attributes[/bold cyan]")
            for attr in attributes:
                attr_branch.add(attr["name"])

        return tree

    def _create_help_panel(self) -> Panel:
        """Create the Rich panel wrapping the display help tree."""
        content = self._create_help_tree()

        title_plain = self._get_help_title()
        title_rich = (
            f"[bold cyan]{title_plain}[/bold cyan]" if title_plain else title_plain
        )

        return Panel(
            content,
            title=title_rich,
            expand=False,
            border_style="orange1",
        )
