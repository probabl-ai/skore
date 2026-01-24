import inspect
import re
import uuid
from abc import ABC, abstractmethod
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from skore._externals._sklearn_compat import parse_version
from skore._utils._environment import is_environment_notebook_like

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
    skore_version = parse_version(version("skore"))
    if skore_version < parse_version("0.1"):
        url_version = "dev"
    else:
        url_version = f"{skore_version.major}.{skore_version.minor}"

    base_url = f"https://docs.skore.probabl.ai/{url_version}/reference/api"
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
# Base help data mixin to extract data from reports, accessors, and displays.
########################################################################################


class _BaseHelpDataMixin(ABC):
    """Base mixin for building help data structures.

    Concrete subclasses implement ``_build_help_data`` for a specific type
    (report, accessor, or display) and reuse these helpers to derive the
    common pieces of information (methods, attributes, descriptions, links).
    """

    @abstractmethod
    def _build_help_data(self) -> dict[str, Any]:
        """Build data structure for Jinja2/Rich rendering."""
        pass

    def _get_methods_for_help(self) -> list[tuple[str, Any]]:
        """Return the public instance methods to display in help."""
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
        return sorted(filtered_methods)

    def _get_method_description(self, method: Any) -> str:
        """Get the one-line description for a method from its docstring."""
        return (
            method.__doc__.split("\n")[0]
            if method.__doc__
            else "No description available"
        )

    def _build_method_data(
        self,
        name: str,
        method: Any,
        obj: Any,
        class_name: str,
        accessor_path: str | None,
    ) -> dict[str, Any]:
        """Build data structure for a single method.

        The method name and parameter list are derived directly from the function
        signature, and the description is taken from the first line of the docstring.
        """
        method_name_only = name
        params_part = ""
        if method is not None:
            try:
                sig = inspect.signature(method)
                param_names = [
                    param_name
                    for param_name, param in sig.parameters.items()
                    if param_name != "self"
                ]
                if param_names:
                    params_part = "(" + ", ".join(param_names) + ")"
                else:
                    params_part = "()"
            except (ValueError, TypeError):
                params_part = "(...)"
        else:
            params_part = "(...)"

        description = obj._get_method_description(method)
        favorability_text = None
        if hasattr(obj, "_get_favorability_text"):
            favorability_text = obj._get_favorability_text(name)

        doc_url = _get_documentation_url(class_name, accessor_path, name)

        return {
            "name_only": method_name_only,
            "params_part": params_part,
            "description": description,
            "favorability_text": favorability_text,
            "doc_url": doc_url,
        }

    def _get_attributes_for_help(self) -> list[str]:
        """Get the attributes ending with '_' to display in help."""
        return sorted(
            name
            for name in dir(self)
            if name.endswith("_") and not name.startswith("_")
        )

    def _get_attribute_description(self, name: str) -> str:
        """Get the description of an attribute from its class docstring."""
        if self.__doc__ is None:
            return "No description available"
        regex_pattern = rf"{name} : .*?\n\s*(.*?)\."
        search_result = re.search(regex_pattern, self.__doc__)
        return search_result.group(1) if search_result else "No description available"

    def _build_attributes_data(
        self, class_name: str
    ) -> tuple[list[dict[str, Any]] | None, dict[str, str] | None]:
        """Build attribute metadata and section identifiers.

        This helper is shared between reports and displays. It assumes that
        `_get_attributes_for_help` returns attribute names without a leading dot.
        """
        attributes = self._get_attributes_for_help()
        if not attributes:
            return None, None

        section = {
            "id": str(uuid.uuid4()),
            "folder_id": str(uuid.uuid4()),
        }

        attrs_without_underscore = [a for a in attributes if not a.endswith("_")]
        attrs_with_underscore = [a for a in attributes if a.endswith("_")]

        items: list[dict[str, Any]] = []
        for attr_name in attrs_without_underscore + attrs_with_underscore:
            description = self._get_attribute_description(attr_name)
            doc_url = _get_documentation_url(class_name, None, attr_name)
            items.append(
                {
                    "name": attr_name,
                    "description": description,
                    "doc_url": doc_url,
                }
            )

        return items, section


class _ReportHelpDataMixin(_BaseHelpDataMixin):
    """Mixin responsible for building help data structures for reports.

    It enriches the generic helpers in ``_BaseHelpDataMixin`` with report-specific
    concepts such as accessors and X/y attributes.
    """

    def _get_attributes_for_help(self) -> list[str]:
        """Get the public attributes to display in help."""
        from skore._sklearn._base import _BaseAccessor

        attributes = []
        xy_attributes = []

        for name in dir(self):
            if (
                name.startswith("_")
                or callable(getattr(self, name))
                or isinstance(getattr(self, name), _BaseAccessor)
            ):
                continue

            value = getattr(self, name)
            if name.startswith(("X", "y")):
                if value is not None:
                    xy_attributes.append(name)
            else:
                attributes.append(name)

        xy_attributes.sort()
        attributes.sort()

        return xy_attributes + attributes

    def _build_help_data(self) -> dict[str, Any]:
        """Build data structure for Jinja2/Rich rendering."""
        title = self._get_help_title()
        class_name = self.__class__.__name__

        data: dict[str, Any] = {
            "title": title,
            "is_report": True,
        }

        data["class_name"] = class_name
        data["accessors"] = []
        data["base_methods"] = []
        data["attributes"] = None
        data["methods_section"] = None
        data["attributes_section"] = None

        for accessor_attr, config in getattr(self, "_ACCESSOR_CONFIG", {}).items():
            accessor = getattr(self, accessor_attr)
            accessor_data = {
                "id": str(uuid.uuid4()),
                "folder_id": str(uuid.uuid4()),
                "name": config["name"],
                "methods": [],
                "sub_accessors": [],
            }

            methods = sorted(accessor._get_methods_for_help())

            for name, method in methods:
                method_data = self._build_method_data(
                    name, method, accessor, class_name, config["name"]
                )
                accessor_data["methods"].append(method_data)

            for sub_attr, sub_obj in inspect.getmembers(accessor):
                from skore._sklearn._base import _BaseAccessor

                if isinstance(sub_obj, _BaseAccessor) and not sub_attr.startswith("_"):
                    sub_accessor_data = {
                        "id": str(uuid.uuid4()),
                        "folder_id": str(uuid.uuid4()),
                        "name": sub_attr,
                        "methods": [],
                    }

                    sub_methods = sorted(sub_obj._get_methods_for_help())

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

        base_methods = sorted(self._get_methods_for_help())

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

        attributes, attributes_section = self._build_attributes_data(class_name)
        data["attributes"] = attributes
        data["attributes_section"] = attributes_section

        return data


class _AccessorHelpDataMixin(_BaseHelpDataMixin):
    """Mixin responsible for building help data structures for accessors."""

    def _build_help_data(self) -> dict[str, Any]:
        """Build data structure for Jinja2/Rich rendering for accessors."""
        title = self._get_help_title()
        class_name = self.__class__.__name__

        data: dict[str, Any] = {
            "title": title,
            "is_report": False,
        }

        if hasattr(self, "_get_help_tree_title"):
            tree_title = self._get_help_tree_title()
        else:
            tree_title = class_name
        data["tree_title"] = tree_title
        data["methods"] = []

        methods = sorted(self._get_methods_for_help())

        for name, method in methods:
            method_data = self._build_method_data(name, method, self, class_name, None)
            data["methods"].append(method_data)

        return data


class _DisplayHelpDataMixin(_BaseHelpDataMixin):
    """Mixin responsible for building help data structures for displays."""

    def _build_help_data(self) -> dict[str, Any]:
        """Build data structure for Jinja2/Rich rendering for displays."""
        class_name = self.__class__.__name__
        title = self._get_help_title()

        data: dict[str, Any] = {
            "title": title,
            "class_name": class_name,
        }

        attributes, attributes_section = self._build_attributes_data(class_name)
        data["attributes"] = attributes
        data["attributes_section"] = attributes_section

        methods = self._get_methods_for_help()
        if methods:
            data["methods_section"] = {
                "id": str(uuid.uuid4()),
                "folder_id": str(uuid.uuid4()),
            }
            data["methods"] = []
            for name, method in methods:
                method_data = self._build_method_data(
                    name=name,
                    method=method,
                    obj=self,
                    class_name=class_name,
                    accessor_path=None,
                )
                data["methods"].append(method_data)
        else:
            data["methods"] = None
            data["methods_section"] = None

        return data


########################################################################################
# Base help mixins for Rich and HTML rendering
########################################################################################


class _BaseRichHelpMixin(ABC):
    """Base mixin for Rich-based help rendering."""

    @abstractmethod
    def _create_help_tree(self) -> Tree:
        """Create the help tree for Rich rendering."""
        pass

    @abstractmethod
    def _create_help_panel(self) -> Panel:
        """Create the Rich panel wrapping the help tree."""
        pass

class _BaseHTMLHelpMixin(ABC):
    """Base mixin for HTML-based help rendering."""

    @abstractmethod
    def _create_help_html(self) -> str:
        """Create the HTML representation of the help tree."""
        pass

########################################################################################
# Report help mixins
########################################################################################


class _RichReportHelpMixin(_ReportHelpDataMixin):
    """Mixin for Rich-based help rendering for reports."""

    def _create_help_tree(self) -> Tree:
        """Create the help tree for Rich rendering (reports only)."""
        data = self._build_help_data()

        tree = Tree(self.__class__.__name__)

        for accessor_data in data.get("accessors", []):
            branch = tree.add(f"[bold cyan].{accessor_data['name']}[/bold cyan]")

            for method in accessor_data["methods"]:
                displayed_name = f"{method['name_only']}(...)"
                description = method["description"]
                branch.add(f".{displayed_name.ljust(25)} - {description}")

            for sub_accessor in accessor_data["sub_accessors"]:
                sub_branch = branch.add(
                    f"[bold cyan].{sub_accessor['name']}[/bold cyan]"
                )
                for method in sub_accessor["methods"]:
                    displayed_name = f"{method['name_only']}(...)"
                    description = method["description"]
                    sub_branch.add(f".{displayed_name.ljust(25)} - {description}")

        base_methods = data.get("base_methods", [])
        if base_methods:
            methods_branch = tree.add("[bold cyan]Methods[/bold cyan]")
            for method in base_methods:
                displayed_name = f"{method['name_only']}(...)"
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


class _HTMLReportHelpMixin(_ReportHelpDataMixin):
    """Mixin for HTML-based help rendering for reports with Shadow DOM isolation."""

    def _create_help_html(self) -> str:
        """Create the HTML representation of the help tree."""
        template_data = self._build_help_data()

        env = _get_jinja_env()
        template = env.get_template("report_help.html.j2")

        container_id = f"skore-help-{uuid.uuid4().hex[:8]}"

        shadow_dom_html = template.render(
            container_id=container_id,
            **template_data,
        )

        return shadow_dom_html


########################################################################################
# Accessor help data & mixins (reports)
########################################################################################


class _RichAccessorHelpMixin(_AccessorHelpDataMixin):
    """Mixin for Rich-based help rendering for accessors."""

    def _create_help_tree(self) -> Tree:
        """Create the help tree for Rich rendering for accessors."""
        data = self._build_help_data()

        tree_title_plain = data.get("tree_title", self.__class__.__name__)
        tree_title_rich = f"[bold cyan]{tree_title_plain}[/bold cyan]"
        tree = Tree(tree_title_rich)

        for method in data.get("methods", []):
            displayed_name = f"{method['name_only']}(...)"
            description = method["description"]
            tree.add(f".{displayed_name}".ljust(26) + f" - {description}")

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


class _HTMLAccessorHelpMixin(_AccessorHelpDataMixin):
    """Mixin for HTML-based help rendering for accessors with Shadow DOM isolation."""

    def _create_help_html(self) -> str:
        """Create the HTML representation of the help tree for accessors."""
        template_data = self._build_help_data()

        env = _get_jinja_env()
        template = env.get_template("report_help.html.j2")

        container_id = f"skore-accessor-help-{uuid.uuid4().hex[:8]}"

        shadow_dom_html = template.render(
            container_id=container_id,
            **template_data,
        )

        return shadow_dom_html


class _RichHelpDisplayMixin(_DisplayHelpDataMixin):
    """Mixin for Rich-based help rendering for displays."""

    def _create_help_tree(self) -> Tree:
        """Create a rich Tree with attributes and methods."""
        data = self._build_help_data()

        tree = Tree("display")

        attributes = data.get("attributes") or []
        if attributes:
            attr_branch = tree.add("[bold cyan]Attributes[/bold cyan]")
            for attr in attributes:
                attr_branch.add(attr["name"])

        methods = data.get("methods") or []
        if methods:
            method_branch = tree.add("[bold cyan]Methods[/bold cyan]")
            for method in methods:
                name = f".{method['name_only']}(...)"
                description = method["description"]
                method_branch.add(f"{name.ljust(26)} - {description}")

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


class _HTMLHelpDisplayMixin(_DisplayHelpDataMixin):
    """Mixin for HTML-based help rendering for displays with Shadow DOM isolation."""

    def _create_help_html(self) -> str:
        """Create the HTML representation of the help tree."""
        template_data = self._build_help_data()

        env = _get_jinja_env()
        template = env.get_template("display_help.html.j2")

        container_id = f"skore-display-help-{uuid.uuid4().hex[:8]}"

        shadow_dom_html = template.render(
            container_id=container_id,
            **template_data,
        )

        return shadow_dom_html


########################################################################################
# Combined help mixins
########################################################################################


class ReportHelpMixin(_RichReportHelpMixin, _HTMLReportHelpMixin):
    """Mixin class providing help for report `help` and `__repr__`.

    This mixin inherits from both `_RichHelpMixin` and `_HTMLHelpMixin` and
    delegates to the appropriate implementation based on the environment.
    """

    def help(self) -> None:
        """Display report help using rich or HTML."""
        if is_environment_notebook_like():
            from IPython.display import HTML, display

            display(HTML(self._create_help_html()))
        else:
            from skore import console  # avoid circular import

            console.print(self._create_help_panel())


class AccessorHelpMixin(_RichAccessorHelpMixin, _HTMLAccessorHelpMixin):
    """Mixin class providing help for accessor `help`."""

    def _get_help_title(self) -> str:
        name = self.__class__.__name__.replace("_", "").replace("Accessor", "").lower()
        return f"{name.capitalize()} accessor"

    def help(self) -> None:
        """Display accessor help using rich or HTML."""
        if is_environment_notebook_like():
            from IPython.display import HTML, display

            display(HTML(self._create_help_html()))
        else:
            from skore import console  # avoid circular import

            console.print(self._create_help_panel())


class DisplayHelpMixin(_RichHelpDisplayMixin, _HTMLHelpDisplayMixin):
    """Mixin class providing help for display `help` and `__repr__`.

    This mixin inherits from both `_RichHelpDisplayMixin` and `_HTMLHelpDisplayMixin`
    and delegates to the appropriate implementation based on the environment.
    """

    estimator_name: str  # defined in the concrete display class

    def _get_help_title(self) -> str:
        """Get the help title for the display."""
        return f"{self.__class__.__name__} display"

    def help(self) -> None:
        """Display display help using rich or HTML."""
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
