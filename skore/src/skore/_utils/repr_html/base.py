import inspect
import re
import uuid
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from skore._utils._environment import is_environment_notebook_like


class ReprHTMLMixin:
    """Mixin to handle consistently the HTML representation.

    When inheriting from this class, you need to define an attribute `_html_repr`
    which is a callable that returns the HTML representation to be shown.
    """

    @property
    def _repr_html_(self):
        return self._repr_html_inner

    def _repr_html_inner(self):
        return self._html_repr()

    def _repr_mimebundle_(self, **kwargs):
        return {"text/plain": repr(self), "text/html": self._html_repr()}


########################################################################################
# Utility functions for help system
########################################################################################


def _get_documentation_url(
    class_name: str,
    accessor_name: str | None = None,
    method_or_attr_name: str | None = None,
) -> str:
    """Generate documentation URL for a method or attribute.

    Parameters
    ----------
    class_name : str
        The class name (e.g., "EstimatorReport", "CrossValidationReport", "ROCCurveDisplay")
    accessor_name : str, optional
        The accessor name if applicable (e.g., "data", "metrics"). Only used for reports.
    method_or_attr_name : str, optional
        The method or attribute name

    Returns
    -------
    str
        The full documentation URL
    """
    base_url = "https://docs.skore.probabl.ai/0.11/reference/api"
    path_parts = ["skore", class_name]

    if accessor_name:
        path_parts.append(accessor_name)

    if method_or_attr_name:
        path_parts.append(method_or_attr_name)

    return f"{base_url}/{'.'.join(path_parts)}.html"


def _get_jinja_env():
    """Get Jinja2 environment for loading templates."""
    from skore._utils import repr_html

    template_dir = Path(repr_html.__file__).parent
    return Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)


########################################################################################
# Rich help mixins
########################################################################################


class _RichHelpMixin:
    """Mixin for Rich-based help rendering for reports."""

    def _create_help_tree(self) -> Tree:
        """Create the help tree for Rich rendering."""
        # Check if this is a report (has _ACCESSOR_CONFIG) or an accessor
        if hasattr(self, "_ACCESSOR_CONFIG"):
            # Report implementation
            tree = Tree(self.__class__.__name__)

            # Add accessor methods first
            for accessor_attr, config in self._ACCESSOR_CONFIG.items():
                accessor = getattr(self, accessor_attr)
                # Add Rich markup for display (plain text is in config['name'])
                branch = tree.add(f"[bold cyan].{config['name']}[/bold cyan]")

                # Add main accessor methods first
                methods = accessor._get_methods_for_help()
                methods = accessor._sort_methods_for_help(methods)

                # Add methods
                for name, method in methods:
                    # For Rich, show only method name with ellipsis
                    displayed_name = f"{name}(...)"
                    description = accessor._get_method_description(method)
                    branch.add(f".{displayed_name.ljust(25)} - {description}")

                # Add sub-accessors after main methods
                for sub_attr, sub_obj in inspect.getmembers(accessor):
                    # Import here to avoid circular dependency
                    from skore._sklearn._base import _BaseAccessor

                    if isinstance(sub_obj, _BaseAccessor) and not sub_attr.startswith(
                        "_"
                    ):
                        sub_branch = branch.add(f"[bold cyan].{sub_attr}[/bold cyan]")

                        # Add sub-accessor methods
                        sub_methods = sub_obj._get_methods_for_help()
                        sub_methods = sub_obj._sort_methods_for_help(sub_methods)

                        for name, method in sub_methods:
                            # For Rich, show only method name with ellipsis
                            displayed_name = f"{name}(...)"
                            description = sub_obj._get_method_description(method)
                            sub_branch.add(
                                f".{displayed_name.ljust(25)} - {description}"
                            )

            # Add base methods section
            base_methods = self._get_methods_for_help()
            base_methods = self._sort_methods_for_help(base_methods)

            if base_methods:
                # Add Rich markup for display (plain text is "Methods")
                methods_branch = tree.add("[bold cyan]Methods[/bold cyan]")
                for name, method in base_methods:
                    description = self._get_method_description(method)
                    # For Rich, show only method name with ellipsis
                    displayed_name = f"{name}(...)"
                    methods_branch.add(
                        f".{displayed_name}".ljust(34) + f" - {description}"
                    )

            # Add attributes section
            if hasattr(self, "_get_attributes_for_help"):
                attributes = self._get_attributes_for_help()
                if attributes:
                    # Add Rich markup for display (plain text is "Attributes")
                    attr_branch = tree.add("[bold cyan]Attributes[/bold cyan]")
                    # Group attributes: those without `_` first, then those with `_`
                    attrs_without_underscore = [
                        a for a in attributes if not a.endswith("_")
                    ]
                    attrs_with_underscore = [a for a in attributes if a.endswith("_")]
                    for attr_name in attrs_without_underscore + attrs_with_underscore:
                        description = self._get_attribute_description(attr_name)
                        attr_branch.add(f".{attr_name.ljust(29)} - {description}")

            return tree
        else:
            # Accessor implementation
            # Get plain text title
            if hasattr(self, "_get_help_tree_title"):
                tree_title_plain = self._get_help_tree_title()
            else:
                tree_title_plain = self.__class__.__name__
            # Add Rich markup for display
            tree_title_rich = f"[bold cyan]{tree_title_plain}[/bold cyan]"
            tree = Tree(tree_title_rich)

            methods = self._get_methods_for_help()
            methods = self._sort_methods_for_help(methods)

            for name, method in methods:
                # For Rich, show only method name with ellipsis
                displayed_name = f"{name}(...)"
                description = self._get_method_description(method)
                tree.add(f".{displayed_name}".ljust(26) + f" - {description}")

            return tree

    def _create_help_panel(self) -> Panel:
        """Create the help panel for Rich rendering."""
        # Legend removed - favorability info now in HTML tooltips only
        content = self._create_help_tree()

        # Get plain text title and add Rich markup for display
        title_plain = self._get_help_panel_title()
        # Add Rich markup for display (if title is not empty)
        title_rich = (
            f"[bold cyan]{title_plain}[/bold cyan]" if title_plain else title_plain
        )

        return Panel(
            content,
            title=title_rich,
            expand=False,
            border_style="orange1",
        )


class _RichHelpDisplayMixin:
    """Mixin for Rich-based help rendering for displays."""

    def _get_attributes_for_help(self) -> list[str]:
        """Get the attributes ending with '_' to display in help."""
        return sorted(
            f".{name}"
            for name in dir(self)
            if name.endswith("_") and not name.startswith("_")
        )

    def _get_methods_for_help(self) -> list[tuple[str, Any]]:
        """Get the public methods to display in help."""
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        filtered_methods = []
        for name, method in methods:
            is_private = name.startswith("_")
            is_class_method = inspect.ismethod(method) and method.__self__ is type(self)
            is_help_method = name == "help"
            if not (is_private or is_class_method or is_help_method):
                filtered_methods.append((f".{name}(...)", method))
        return sorted(filtered_methods)

    def _create_help_tree(self) -> Tree:
        """Create a rich Tree with attributes and methods."""
        tree = Tree("display")

        methods = self._get_methods_for_help()
        method_branch = tree.add("[bold cyan]Methods[/bold cyan]")
        for name, method in methods:
            description = (
                method.__doc__.split("\n")[0]
                if method.__doc__
                else "No description available"
            )
            method_branch.add(f"{name} - {description}")

        attributes = self._get_attributes_for_help()
        # Ensure figure_ and ax_ are first
        sorted_attrs = sorted(attributes)
        if (".figure_" in sorted_attrs) and (".ax_" in sorted_attrs):
            sorted_attrs.remove(".ax_")
            sorted_attrs.remove(".figure_")
            sorted_attrs = [".figure_", ".ax_"] + [
                attr for attr in sorted_attrs if attr not in [".figure_", ".ax_"]
            ]
        if sorted_attrs:
            attr_branch = tree.add("[bold cyan]Attributes[/bold cyan]")
            for attr in sorted_attrs:
                attr_branch.add(attr)

        return tree

    def _create_help_panel(self) -> Panel:
        return Panel(
            self._create_help_tree(),
            title=f"[bold cyan]{self.__class__.__name__} [/bold cyan]",
            border_style="orange1",
            expand=False,
        )


########################################################################################
# HTML help mixins
########################################################################################


class _HTMLHelpMixin:
    """Mixin for HTML-based help rendering for reports with Shadow DOM isolation."""

    def _build_help_data(self) -> dict[str, Any]:
        """Build data structure for Jinja2 template rendering."""
        # Get plain text title (base methods should return plain text)
        title = self._get_help_panel_title()
        class_name = self.__class__.__name__

        data: dict[str, Any] = {
            "title": title,
            "is_report": hasattr(self, "_ACCESSOR_CONFIG"),
        }

        if data["is_report"]:
            # Report implementation
            data["class_name"] = class_name
            data["accessors"] = []
            data["base_methods"] = []
            data["attributes"] = None
            data["methods_section"] = None
            data["attributes_section"] = None

            # Build accessors data
            for accessor_attr, config in self._ACCESSOR_CONFIG.items():
                accessor = getattr(self, accessor_attr)
                accessor_data = {
                    "id": str(uuid.uuid4()),
                    "folder_id": str(uuid.uuid4()),
                    "name": config["name"],
                    "methods": [],
                    "sub_accessors": [],
                }

                # Add main accessor methods
                methods = accessor._get_methods_for_help()
                methods = accessor._sort_methods_for_help(methods)

                for name, method in methods:
                    method_data = self._build_method_data(
                        name, method, accessor, class_name, config["name"]
                    )
                    accessor_data["methods"].append(method_data)

                # Add sub-accessors
                for sub_attr, sub_obj in inspect.getmembers(accessor):
                    # Import here to avoid circular dependency
                    from skore._sklearn._base import _BaseAccessor

                    if isinstance(sub_obj, _BaseAccessor) and not sub_attr.startswith(
                        "_"
                    ):
                        sub_accessor_data = {
                            "id": str(uuid.uuid4()),
                            "folder_id": str(uuid.uuid4()),
                            "name": sub_attr,
                            "methods": [],
                        }

                        sub_methods = sub_obj._get_methods_for_help()
                        sub_methods = sub_obj._sort_methods_for_help(sub_methods)

                        for name, method in sub_methods:
                            method_data = self._build_method_data(
                                name,
                                method,
                                sub_obj,
                                class_name,
                                f"{config['name']}.{sub_attr}",
                            )
                            sub_accessor_data["methods"].append(method_data)

                        accessor_data["sub_accessors"].append(sub_accessor_data)

                data["accessors"].append(accessor_data)

            # Build base methods data
            base_methods = self._get_methods_for_help()
            base_methods = self._sort_methods_for_help(base_methods)

            if base_methods:
                data["methods_section"] = {
                    "id": str(uuid.uuid4()),
                    "folder_id": str(uuid.uuid4()),
                }
                for name, method in base_methods:
                    method_data = self._build_method_data(
                        name, method, self, class_name, None
                    )
                    data["base_methods"].append(method_data)

            # Build attributes data
            if hasattr(self, "_get_attributes_for_help"):
                attributes = self._get_attributes_for_help()
                if attributes:
                    data["attributes_section"] = {
                        "id": str(uuid.uuid4()),
                        "folder_id": str(uuid.uuid4()),
                    }
                    attrs_without_underscore = [
                        a for a in attributes if not a.endswith("_")
                    ]
                    attrs_with_underscore = [a for a in attributes if a.endswith("_")]
                    data["attributes"] = []
                    for attr_name in attrs_without_underscore + attrs_with_underscore:
                        description = self._get_attribute_description(attr_name)
                        # Descriptions are already plain text from docstrings
                        doc_url = _get_documentation_url(class_name, None, attr_name)
                        data["attributes"].append(
                            {
                                "name": attr_name,
                                "description": description,
                                "doc_url": doc_url,
                            }
                        )
        else:
            # Accessor implementation
            # Get plain text title (base methods should return plain text)
            if hasattr(self, "_get_help_tree_title"):
                tree_title = self._get_help_tree_title()
            else:
                tree_title = self.__class__.__name__
            data["tree_title"] = tree_title
            data["methods"] = []

            methods = self._get_methods_for_help()
            methods = self._sort_methods_for_help(methods)

            for name, method in methods:
                method_data = self._build_method_data(
                    name, method, self, class_name, None
                )
                data["methods"].append(method_data)

        return data

    def _build_method_data(
        self,
        name: str,
        method: Any,
        obj: Any,
        class_name: str,
        accessor_path: str | None,
    ) -> dict[str, Any]:
        """Build data structure for a single method.

        Note: This method works with plain text. Method names and descriptions
        from _format_method_name and _get_method_description are already plain text.
        """
        displayed_name = obj._format_method_name(name, method)
        description = obj._get_method_description(method)
        # Both displayed_name and description are already plain text (no Rich markup)

        # Split method name from parameters
        if "(" in displayed_name:
            method_name_only, params_part = displayed_name.split("(", 1)
            params_part = "(" + params_part
        else:
            method_name_only = displayed_name
            params_part = ""

        # Get favorability text if applicable
        favorability_text = None
        if hasattr(obj, "_get_favorability_text"):
            favorability_text = obj._get_favorability_text(name)

        # Generate documentation URL
        doc_url = _get_documentation_url(class_name, accessor_path, name)

        return {
            "name_only": method_name_only,
            "params_part": params_part,
            "description": description,
            "favorability_text": favorability_text,
            "doc_url": doc_url,
        }

    def _create_help_html(self) -> str:
        """Create the HTML representation of the help tree."""
        # Build data structure for template
        template_data = self._build_help_data()

        # Load template
        env = _get_jinja_env()
        template = env.get_template("report_help.html.j2")

        # Generate unique ID for this instance
        container_id = f"skore-help-{uuid.uuid4().hex[:8]}"

        # Render the template with all data
        # CSS and JS are included directly via {% include %} in the template
        shadow_dom_html = template.render(
            container_id=container_id,
            **template_data,
        )

        return shadow_dom_html


class _HTMLHelpDisplayMixin:
    """Mixin for HTML-based help rendering for displays with Shadow DOM isolation."""

    def _get_attribute_description(self, name: str) -> str:
        """Get the description of an attribute from its class docstring."""
        if self.__doc__ is None:
            return "No description available"
        # Look for the first sentence on the line below the pattern 'attribute_name : '
        regex_pattern = rf"{name} : .*?\n\s*(.*?)\."
        search_result = re.search(regex_pattern, self.__doc__)
        return search_result.group(1) if search_result else "No description available"

    def _build_help_data(self) -> dict[str, Any]:
        """Build data structure for Jinja2 template rendering."""
        class_name = self.__class__.__name__
        title = class_name

        data: dict[str, Any] = {
            "title": title,
            "class_name": class_name,
        }

        # Build attributes data
        attributes = self._get_attributes_for_help()
        if attributes:
            data["attributes_section"] = {
                "id": str(uuid.uuid4()),
                "folder_id": str(uuid.uuid4()),
            }
            # Ensure figure_ and ax_ are first
            sorted_attrs = sorted(attributes)
            if (".figure_" in sorted_attrs) and (".ax_" in sorted_attrs):
                sorted_attrs.remove(".ax_")
                sorted_attrs.remove(".figure_")
                sorted_attrs = [".figure_", ".ax_"] + [
                    attr for attr in sorted_attrs if attr not in [".figure_", ".ax_"]
                ]
            data["attributes"] = []
            for attr_name in sorted_attrs:
                # Remove the leading dot for the name
                attr_name_clean = attr_name.lstrip(".")
                description = self._get_attribute_description(attr_name_clean)
                doc_url = _get_documentation_url(class_name, None, attr_name_clean)
                data["attributes"].append(
                    {
                        "name": attr_name,
                        "description": description,
                        "doc_url": doc_url,
                    }
                )
        else:
            data["attributes"] = None
            data["attributes_section"] = None

        # Build methods data
        methods = self._get_methods_for_help()
        if methods:
            data["methods_section"] = {
                "id": str(uuid.uuid4()),
                "folder_id": str(uuid.uuid4()),
            }
            data["methods"] = []
            for name, method in methods:
                # Remove the leading dot and (...)
                method_name_clean = name.lstrip(".").replace("(...)", "")
                description = (
                    method.__doc__.split("\n")[0]
                    if method.__doc__
                    else "No description available"
                )
                doc_url = _get_documentation_url(class_name, None, method_name_clean)
                # Split method name from parameters
                if "(" in name:
                    method_name_only, params_part = name.split("(", 1)
                    params_part = "(" + params_part
                else:
                    method_name_only = name
                    params_part = ""
                data["methods"].append(
                    {
                        "name_only": method_name_only,
                        "params_part": params_part,
                        "description": description,
                        "doc_url": doc_url,
                    }
                )
        else:
            data["methods"] = None
            data["methods_section"] = None

        return data

    def _create_help_html(self) -> str:
        """Create the HTML representation of the help tree."""
        # Build data structure for template
        template_data = self._build_help_data()

        # Load template
        env = _get_jinja_env()
        template = env.get_template("display_help.html.j2")

        # Generate unique ID for this instance
        container_id = f"skore-display-help-{uuid.uuid4().hex[:8]}"

        # Render the template with all data
        # CSS and JS are included directly via {% include %} in the template
        shadow_dom_html = template.render(
            container_id=container_id,
            **template_data,
        )

        return shadow_dom_html


########################################################################################
# Combined help mixins
########################################################################################


class _HelpMixin(_RichHelpMixin, _HTMLHelpMixin, ABC):
    """Mixin class providing help for the `help` method and the `__repr__` method.

    This mixin inherits from both `_RichHelpMixin` and `_HTMLHelpMixin` and
    delegates to the appropriate implementation based on the environment.
    """

    def _get_methods_for_help(self) -> list[tuple[str, Any]]:
        """Get the methods to display in help."""
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        filtered_methods = []
        for name, method in methods:
            is_private_method = name.startswith("_")
            # we cannot use `isinstance(method, classmethod)` because it is already
            # transformed by the decorator `@classmethod`.
            is_class_method = inspect.ismethod(method) and method.__self__ is type(self)
            is_help_method = name == "help"
            if not (is_private_method or is_class_method or is_help_method):
                filtered_methods.append((name, method))
        return filtered_methods

    def _sort_methods_for_help(
        self, methods: list[tuple[str, Any]]
    ) -> list[tuple[str, Any]]:
        """Sort methods for help display."""
        return sorted(methods)

    def _format_method_name(self, name: str, method: Any | None = None) -> str:
        """Format method name for display with actual parameter signature.

        Parameters
        ----------
        name : str
            The name of the method.
        method : Any, optional
            The method object. If not provided, will be retrieved using getattr.
        """
        if method is None:
            method = getattr(self, name, None)

        if method is None:
            return f"{name}(...)"

        try:
            sig = inspect.signature(method)
            # Get parameter string, removing 'self' parameter
            params = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                params.append(param_name)

            params_str = ", ".join(params)
            full_signature = f"{name}({params_str})"

            return full_signature
        except (ValueError, TypeError):
            # Fallback to ellipsis if signature cannot be obtained
            return f"{name}(...)"

    def _get_method_description(self, method: Any) -> str:
        """Get the description for a method."""
        return (
            method.__doc__.split("\n")[0]
            if method.__doc__
            else "No description available"
        )

    def _get_attribute_description(self, name: str) -> str:
        """Get the description of an attribute from its class docstring."""
        if self.__doc__ is None:
            return "No description available"
        # Look for the first sentence on the line below the pattern 'attribute_name : '
        regex_pattern = rf"{name} : .*?\n\s*(.*?)\."
        search_result = re.search(regex_pattern, self.__doc__)
        return search_result.group(1) if search_result else "No description available"

    @abstractmethod
    def _get_help_panel_title(self) -> str:
        """Get the help panel title."""

    def help(self) -> None:
        """Display a rich help."""
        if is_environment_notebook_like():
            from IPython.display import HTML, display

            display(HTML(self._create_help_html()))
        else:
            from skore import console  # avoid circular import

            console.print(self._create_help_panel())

    def _rich_repr(self, class_name: str) -> str:
        """Return a string representation using rich."""
        string_buffer = StringIO()
        console = Console(file=string_buffer, force_terminal=False)
        console.print(
            Panel(
                "Get guidance using the help() method",
                title=f"[cyan]{class_name}[/cyan]",
                border_style="orange1",
                expand=False,
            )
        )
        return string_buffer.getvalue()


class HelpDisplayMixin(_RichHelpDisplayMixin, _HTMLHelpDisplayMixin):
    """Mixin class providing help for the `help` method and the `__repr__` method.

    This mixin inherits from both `_RichHelpDisplayMixin` and `_HTMLHelpDisplayMixin` and
    delegates to the appropriate implementation based on the environment.
    """

    estimator_name: str  # defined in the concrete display class

    def help(self) -> None:
        """Display available attributes and methods using rich or HTML."""
        if is_environment_notebook_like():
            from IPython.display import HTML, display

            display(HTML(self._create_help_html()))
        else:
            from skore import console  # avoid circular import

            console.print(self._create_help_panel())

    def __str__(self) -> str:
        """Return a string representation using rich."""
        string_buffer = StringIO()
        console = Console(file=string_buffer, force_terminal=False)
        console.print(
            Panel(
                "Get guidance using the help() method",
                title=f"[cyan]{self.__class__.__name__}[/cyan]",
                border_style="orange1",
                expand=False,
            )
        )
        return string_buffer.getvalue()

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        string_buffer = StringIO()
        console = Console(file=string_buffer, force_terminal=False)
        console.print(f"[cyan]skore.{self.__class__.__name__}(...)[/cyan]")
        return string_buffer.getvalue()
