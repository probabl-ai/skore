"""Help data mixins: extract and structure data for reports, accessors, and displays."""

import inspect
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Callable
from urllib.parse import quote

from skore._externals._sklearn_compat import parse_version


@dataclass
class HelpSection:
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
    accessor_name: str
    accessor_branch_id: str
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


def _get_attribute_type(obj: Any, attribute_name: str) -> str | None:
    """Extract the type part from a numpydoc-style attribute or parameter entry.

    Parameters
    ----------
    obj : object
        The instance whose class docstring to search.
    attribute_name : str
        The attribute name (e.g. ``"coef_"``, ``"classes_"``).

    Returns
    -------
    str or None
        The type snippet, or ``None`` if not found.
    """
    if obj.__doc__ is None:
        return None
    pattern = rf"{re.escape(attribute_name)} \: ([^\n]+)"
    match = re.search(pattern, obj.__doc__)
    return match.group(1).strip() if match else None


def _build_attribute_text_fragment(obj: Any, attribute_name: str) -> str:
    """Build the encoded ``#:~:text=`` fragment value for an attribute link.

    Uses ``_get_attribute_type`` to extract the type from the class docstring.
    Encodes name and type separately (spaces, brackets, etc.) and keeps
    ``,-`` literal for the scroll-to-text prefix-,match format. When no type
    is found, returns the encoded attribute name only.

    Parameters
    ----------
    obj : object
        The instance whose class docstring to search for the attribute type.
    attribute_name : str
        The attribute name (e.g. ``"coef_"``, ``"classes_"``).

    Returns
    -------
    str
        The percent-encoded fragment value to append after ``#:~:text=``.
    """
    attribute_type = _get_attribute_type(obj, attribute_name)
    if attribute_type is not None:
        return f"{quote(attribute_name, safe='')},-{quote(attribute_type, safe='')}"
    return quote(attribute_name, safe="")


def get_documentation_url(
    *,
    obj: Any,
    accessor_name: str | None = None,
    method_name: str | None = None,
    attribute_name: str | None = None,
) -> str:
    """Generate documentation URL for a class, a method or an attribute.

    Parameters
    ----------
    obj : object
        The instance whose class defines the documented API (e.g. report,
        display, or accessor's parent report). Must be the report/display
        for attribute URLs so the docstring is searched correctly.
    accessor_name : str, default=None
        The accessor name if applicable (e.g. ``"data"``, ``"metrics"``).
        Only used for reports.
    method_name : str, default=None
        The method name.
    attribute_name : str, default=None
        The attribute name. The text fragment is derived from the class
        docstring when possible.

    Returns
    -------
    str
        The full documentation URL, with optional ``#:~:text=...`` fragment
        for attributes.
    """
    class_name = obj.__class__.__name__
    skore_version = parse_version(version("skore"))
    if skore_version < parse_version("0.1"):
        url_version = "dev"
    else:
        url_version = f"{skore_version.major}.{skore_version.minor}"

    base_url = f"https://docs.skore.probabl.ai/{url_version}/reference/api"
    path_parts = ["skore", class_name]

    if accessor_name is not None:
        path_parts.append(accessor_name)
        if method_name is not None:
            path_parts.append(method_name)

    full_url = f"{base_url}/{'.'.join(path_parts)}.html"

    if attribute_name:
        return (
            f"{full_url}#:~:text={_build_attribute_text_fragment(obj, attribute_name)}"
        )

    if method_name is not None and accessor_name is None:
        path_parts.append(method_name)
        return f"{full_url}#{'.'.join(path_parts)}"

    return full_url


def get_public_methods(obj: Any) -> list[tuple[str, Any]]:
    """Return the public instance methods of ``obj`` to display in help.

    Excludes private methods (leading underscore), class methods, and the
    ``help`` method itself.

    Parameters
    ----------
    obj : object
        The instance to inspect (e.g. a report, accessor, or display).

    Returns
    -------
    list of tuple of (str, callable)
        Pairs of (method name, method) sorted by name.
    """
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
    """Extract the one-line description of a method from its docstring.

    Uses the first line of the docstring; returns a fallback if none exists.

    Parameters
    ----------
    method : callable
        The method whose docstring to read.

    Returns
    -------
    str
        The description of the method.
    """
    return (
        method.__doc__.split("\n")[0] if method.__doc__ else "No description available"
    )


def get_public_attributes(obj: Any) -> list[str]:
    """Return public, non-callable attribute names of ``obj`` for help.

    Excludes names starting with ``_``, callables (e.g. methods), and
    attributes that are ``_BaseAccessor`` instances. Typically includes
    fitted attributes (e.g. ending with ``_``) and other instance data.

    Parameters
    ----------
    obj : object
        The instance to inspect (e.g. a report or display).

    Returns
    -------
    list of str
        Sorted attribute names to display in help.
    """
    from skore._sklearn._base import _BaseAccessor  # avoid circular import

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
    """Extract a short description of an attribute from the class docstring.

    Looks for a numpydoc-style "Attributes" entry ``{name} : type\\n    Description.``
    and returns the description text.

    Parameters
    ----------
    obj : object
        The instance whose class docstring to search.
    name : str
        The attribute name (e.g. ``"coef_"``, ``"classes_"``).

    Returns
    -------
    str
        The extracted description, or ``"No description available"``.
    """
    if obj.__doc__ is None:
        return "No description available"
    regex_pattern = rf"{name} : .*?\n\s*(.*?)\."
    search_result = re.search(regex_pattern, obj.__doc__)
    return search_result.group(1) if search_result else "No description available"


class _BaseHelpDataMixin(ABC):
    """Base mixin for building help data structures.

    Enforces children to implement the following methods:
    - ``_get_help_title``: to get the help title for the report, accessor, or display.
    - ``_build_help_data``: to build the data structure for Jinja2/Rich rendering.

    It provides the concreate implementation to get a method data structure or a list
    of attribute data structures.
    """

    @abstractmethod
    def _get_help_title(self) -> str:
        """Get the help title for the report, accessor, or display."""

    @abstractmethod
    def _build_help_data(self) -> ReportHelpData | AccessorHelpData | DisplayHelpData:
        """Build data structure for Jinja2/Rich rendering."""

    def _build_method_data(
        self,
        *,
        name: str,
        method: Callable,
        obj: Any,
        parent_obj: Any,
        accessor_name: str | None,
    ) -> MethodHelp:
        """Build data structure for a single method.

        The parameter list is derived from the function signature; the description
        comes from the first line of the docstring. Optional favorability text
        is taken from ``obj._get_favorability_text(name)`` when available.
        The documentation URL uses ``parent_obj`` and ``accessor_name`` for
        path construction.

        Parameters
        ----------
        name : str
            The method name.
        method : callable
            The method to inspect for signature and docstring.
        obj : object
            The instance that owns the method (report, accessor, or display).
            Used for favorability text when ``_get_favorability_text`` exists.
        parent_obj : object
            The parent instance for URL construction (report or display).
        accessor_name : str or None
            The accessor name when documenting an accessor method, else ``None``.

        Returns
        -------
        MethodHelp
            Dataclass with ``name``, ``parameters``, ``description``,
            ``favorability``, and ``doc_url``.
        """
        sig = inspect.signature(method)
        parameter_names = [
            parameter_name
            for parameter_name, _ in sig.parameters.items()
            if parameter_name != "self"
        ]
        if parameter_names:
            parameters = "(" + ", ".join(parameter_names) + ")"
        else:
            parameters = "()"

        description = get_method_short_summary(method)
        favorability = None
        if hasattr(obj, "_get_favorability_text"):
            favorability = obj._get_favorability_text(name)

        doc_url = get_documentation_url(
            obj=parent_obj,
            accessor_name=accessor_name,
            method_name=name,
        )

        return MethodHelp(
            name=name,
            parameters=parameters,
            description=description,
            favorability=favorability,
            doc_url=doc_url,
        )

    def _build_attributes_data(
        self,
    ) -> tuple[list[AttributeHelp] | None, HelpSection | None]:
        """Build attribute metadata and section identifiers for help rendering.

        Collects public, non-callable attributes via ``get_public_attributes``,
        orders them (names not ending with ``_`` first, then those ending with
        ``_``), and builds ``AttributeHelp`` entries with descriptions from
        the class docstring and documentation URLs. Returns ``(None, None)``
        when there are no attributes.

        Returns
        -------
        items : list of AttributeHelp or None
            The attribute help entries, or ``None`` if no attributes.
        section : HelpSection or None
            Section branch_id for the attributes block, or ``None``.
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
                    obj=self,
                    attribute_name=attr_name,
                ),
            )
            for attr_name in attrs_without_underscore + attrs_with_underscore
        ]
        section = HelpSection(branch_id=str(uuid.uuid4()))
        return items, section


class _ReportHelpDataMixin(_BaseHelpDataMixin):
    """Mixin responsible for building help data structures for reports.

    It defines ``_build_help_data`` by looking at accessors, methods, and attributes
    of a report.
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]]

    def _build_help_data(self) -> ReportHelpData:
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
                    parent_obj=self,
                    accessor_name=config["name"],
                )
                for name, method in get_public_methods(accessor)
            ]
            accessors.append(
                AccessorBranchHelp(
                    branch_id=str(uuid.uuid4()),
                    name=config["name"],
                    methods=methods,
                )
            )

        base_methods_raw = get_public_methods(self)
        methods_section, base_methods = None, []
        if base_methods_raw:
            methods_section = HelpSection(branch_id=str(uuid.uuid4()))
            base_methods = [
                self._build_method_data(
                    name=name,
                    method=method,
                    obj=self,
                    parent_obj=self,
                    accessor_name=None,
                )
                for name, method in base_methods_raw
            ]

        attributes, attributes_section = self._build_attributes_data()

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
    """Mixin responsible for building help data structures for accessors.

    It defines ``_build_help_data`` by looking at methods of an accessor.
    """

    _verbose_name: str
    _parent: Any

    def _build_help_data(self) -> AccessorHelpData:
        """Build data structure for Jinja2/Rich rendering for accessors."""
        root_node = self._parent.__class__.__name__
        methods = [
            self._build_method_data(
                name=name,
                method=method,
                obj=self,
                parent_obj=self._parent,
                accessor_name=self._verbose_name,
            )
            for name, method in get_public_methods(self)
        ]
        return AccessorHelpData(
            title=self._get_help_title(),
            root_node=root_node,
            accessor_name=self._verbose_name,
            accessor_branch_id=str(uuid.uuid4()),
            methods=methods,
        )


class _DisplayHelpDataMixin(_BaseHelpDataMixin):
    """Mixin responsible for building help data structures for displays.

    It defines ``_build_help_data`` by looking at attributes and methods of a display.
    """

    def _build_help_data(self) -> DisplayHelpData:
        """Build data structure for Jinja2/Rich rendering for displays."""
        class_name = self.__class__.__name__
        title = self._get_help_title()
        attributes, attributes_section = self._build_attributes_data()

        methods_raw = get_public_methods(self)
        methods_section, methods = None, []
        if methods_raw:
            methods_section = HelpSection(branch_id=str(uuid.uuid4()))
            methods = [
                self._build_method_data(
                    name=name,
                    method=method,
                    obj=self,
                    parent_obj=self,
                    accessor_name=None,
                )
                for name, method in methods_raw
            ]

        return DisplayHelpData(
            title=title,
            root_node=class_name,
            class_name=class_name,
            attributes=attributes,
            attributes_section=attributes_section,
            methods_section=methods_section,
            methods=methods,
        )
