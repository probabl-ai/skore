"""Help data mixins: extract and structure data for reports, accessors, and displays."""

import inspect
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

from skore._externals._sklearn_compat import parse_version


@dataclass
class HelpSection:
    id: str
    branch_id: str


@dataclass
class MethodHelp:
    name: str
    parameters: str
    description: str
    favorability: str | None
    doc_url: str


@dataclass
class AttributeHelp:
    name: str
    description: str
    doc_url: str


@dataclass
class AccessorBranchHelp:
    id: str
    branch_id: str
    name: str
    methods: list[MethodHelp]


@dataclass
class ReportHelpData:
    title: str
    root_node: str
    class_name: str
    accessors: list[AccessorBranchHelp]
    base_methods: list[MethodHelp]
    methods_section: HelpSection | None
    attributes: list[AttributeHelp] | None
    attributes_section: HelpSection | None


@dataclass
class AccessorHelpData:
    title: str
    root_node: str
    methods: list[MethodHelp]


@dataclass
class DisplayHelpData:
    title: str
    root_node: str
    class_name: str
    attributes: list[AttributeHelp] | None
    attributes_section: HelpSection | None
    methods_section: HelpSection | None
    methods: list[MethodHelp] | None


def get_documentation_url(
    *,
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


def get_public_methods_for_help(obj: Any) -> list[tuple[str, Any]]:
    """Return the public instance methods of ``obj`` to display in help."""
    methods = inspect.getmembers(obj, predicate=inspect.ismethod)
    filtered_methods = []
    for name, method in methods:
        is_private_method = name.startswith("_")
        # we cannot use `isinstance(method, classmethod)` because it is already
        # transformed by the decorator `@classmethod`.
        is_class_method = inspect.ismethod(method) and method.__self__ is type(obj)
        is_help_method = name == "help"
        if not (is_private_method or is_class_method or is_help_method):
            filtered_methods.append((name, method))
    return sorted(filtered_methods)


def get_method_short_summary(method: Any) -> str:
    """Get the one-line description for a method from its docstring."""
    return (
        method.__doc__.split("\n")[0] if method.__doc__ else "No description available"
    )


def get_public_attributes(obj: Any) -> list[str]:
    """Get the attributes of ``obj`` ending with '_' to display in help."""
    from skore._sklearn._base import _BaseAccessor

    return sorted(
        name
        for name in dir(obj)
        if not (
            name.startswith("_")
            or callable(getattr(obj, name))
            or isinstance(getattr(obj, name), _BaseAccessor)
        )
    )


def get_attribute_short_summary(obj: Any, name: str) -> str:
    """Get the description of an attribute from the class docstring."""
    if obj.__doc__ is None:
        return "No description available"
    regex_pattern = rf"{name} : .*?\n\s*(.*?)\."
    search_result = re.search(regex_pattern, obj.__doc__)
    return search_result.group(1) if search_result else "No description available"


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

    def _build_method_data(
        self,
        *,
        name: str,
        method: Any,
        obj: Any,
        class_name: str,
        accessor_path: str | None,
    ) -> MethodHelp:
        """Build data structure for a single method.

        The method name and parameter list are derived directly from the function
        signature, and the description is taken from the first line of the docstring.
        """
        parameters = ""
        if method is not None:
            try:
                sig = inspect.signature(method)
                param_names = [
                    param_name
                    for param_name, _ in sig.parameters.items()
                    if param_name != "self"
                ]
                if param_names:
                    parameters = "(" + ", ".join(param_names) + ")"
                else:
                    parameters = "()"
            except (ValueError, TypeError):
                parameters = "(...)"
        else:
            parameters = "(...)"

        description = get_method_short_summary(method)
        favorability = None
        if hasattr(obj, "_get_favorability_text"):
            favorability = obj._get_favorability_text(name)

        doc_url = get_documentation_url(
            class_name=class_name,
            accessor_name=accessor_path,
            method_or_attr_name=name,
        )

        return MethodHelp(
            name=name,
            parameters=parameters,
            description=description,
            favorability=favorability,
            doc_url=doc_url,
        )

    def _build_attributes_data(
        self, *, class_name: str
    ) -> tuple[list[AttributeHelp] | None, HelpSection | None]:
        """Build attribute metadata and section identifiers.

        This helper is shared between reports and displays. It assumes that
        `get_public_attributes` returns attribute names without a leading dot.
        """
        attributes = get_public_attributes(self)
        if not attributes:
            return None, None

        attrs_without_underscore = [a for a in attributes if not a.endswith("_")]
        attrs_with_underscore = [a for a in attributes if a.endswith("_")]

        items = [
            AttributeHelp(
                name=attr_name,
                description=get_attribute_short_summary(self, attr_name),
                doc_url=get_documentation_url(
                    class_name=class_name,
                    accessor_name=None,
                    method_or_attr_name=attr_name,
                ),
            )
            for attr_name in attrs_without_underscore + attrs_with_underscore
        ]
        section = HelpSection(id=str(uuid.uuid4()), branch_id=str(uuid.uuid4()))
        return items, section


class _ReportHelpDataMixin(_BaseHelpDataMixin):
    """Mixin responsible for building help data structures for reports.

    It enriches the generic helpers in ``_BaseHelpDataMixin`` with report-specific
    concepts such as accessors and X/y attributes.
    """

    def _build_help_data(self) -> dict[str, Any]:
        """Build data structure for Jinja2/Rich rendering."""
        class_name = self.__class__.__name__
        title = self._get_help_title()

        accessors = []
        for accessor_attr, config in self._ACCESSOR_CONFIG.items():
            accessor = getattr(self, accessor_attr)
            methods = [
                self._build_method_data(
                    name=name,
                    method=method,
                    obj=accessor,
                    class_name=class_name,
                    accessor_path=config["name"],
                )
                for name, method in get_public_methods_for_help(accessor)
            ]
            accessors.append(
                AccessorBranchHelp(
                    id=str(uuid.uuid4()),
                    branch_id=str(uuid.uuid4()),
                    name=config["name"],
                    methods=methods,
                )
            )

        base_methods_raw = get_public_methods_for_help(self)
        if base_methods_raw:
            methods_section = HelpSection(
                id=str(uuid.uuid4()), branch_id=str(uuid.uuid4())
            )
            base_methods = [
                self._build_method_data(
                    name=name,
                    method=method,
                    obj=self,
                    class_name=class_name,
                    accessor_path=None,
                )
                for name, method in base_methods_raw
            ]
        else:
            methods_section = None
            base_methods = []

        attributes, attributes_section = self._build_attributes_data(
            class_name=class_name
        )

        return ReportHelpData(
            title=title,
            root_node=class_name,
            class_name=class_name,
            accessors=accessors,
            base_methods=base_methods,
            methods_section=methods_section,
            attributes=attributes,
            attributes_section=attributes_section,
        )


class _AccessorHelpDataMixin(_BaseHelpDataMixin):
    """Mixin responsible for building help data structures for accessors."""

    def _build_help_data(self) -> dict[str, Any]:
        """Build data structure for Jinja2/Rich rendering for accessors."""
        class_name = self.__class__.__name__
        root_node = f"{self._parent.__class__.__name__}.{self._verbose_name}"
        methods = [
            self._build_method_data(
                name=name,
                method=method,
                obj=self,
                class_name=class_name,
                accessor_path=None,
            )
            for name, method in get_public_methods_for_help(self)
        ]
        return AccessorHelpData(
            title=self._get_help_title(),
            root_node=root_node,
            methods=methods,
        )


class _DisplayHelpDataMixin(_BaseHelpDataMixin):
    """Mixin responsible for building help data structures for displays."""

    def _build_help_data(self) -> dict[str, Any]:
        """Build data structure for Jinja2/Rich rendering for displays."""
        class_name = self.__class__.__name__
        title = self._get_help_title()
        attributes, attributes_section = self._build_attributes_data(
            class_name=class_name
        )

        methods_raw = get_public_methods_for_help(self)
        if methods_raw:
            methods_section = HelpSection(
                id=str(uuid.uuid4()), branch_id=str(uuid.uuid4())
            )
            methods = [
                self._build_method_data(
                    name=name,
                    method=method,
                    obj=self,
                    class_name=class_name,
                    accessor_path=None,
                )
                for name, method in methods_raw
            ]
        else:
            methods_section = None
            methods = None

        return DisplayHelpData(
            title=title,
            root_node=class_name,
            class_name=class_name,
            attributes=attributes,
            attributes_section=attributes_section,
            methods_section=methods_section,
            methods=methods,
        )
