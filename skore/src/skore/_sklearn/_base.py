import inspect
import re
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, cast

import joblib
from jinja2 import Environment, FileSystemLoader
from numpy.typing import ArrayLike, NDArray
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from sklearn.base import BaseEstimator
from sklearn.utils._response import _check_response_method, _get_response_values

from skore._externals._sklearn_compat import is_clusterer
from skore._sklearn.types import PositiveLabel
from skore._utils._cache import Cache
from skore._utils._measure_time import MeasureTime


def _get_documentation_url(
    class_name: str,
    accessor_name: str | None = None,
    method_or_attr_name: str | None = None,
) -> str:
    """Generate documentation URL for a method or attribute.

    Parameters
    ----------
    class_name : str
        The class name (e.g., "EstimatorReport", "CrossValidationReport")
    accessor_name : str, optional
        The accessor name if applicable (e.g., "data", "metrics")
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


def _strip_rich_markup(text: str) -> str:
    """Strip all Rich markup tags from a string.

    Rich markup uses the format [tag]content[/tag]. This function removes
    all such tags including self-closing tags.
    """
    # Remove all Rich markup tags: [tag] or [/tag] or [tag]content[/tag]
    # This regex handles:
    # - Opening tags: [tag]
    # - Closing tags: [/tag]
    # - Self-contained tags: [tag]content[/tag]
    # We need to handle nested brackets and escaped brackets
    while True:
        # Remove Rich markup tags - [anything] or [/anything]
        new_text = re.sub(r"\[/?[^\]]*\]", "", text)
        if new_text == text:
            break
        text = new_text
    return text


def _get_jinja_env():
    """Get Jinja2 environment for loading templates."""
    from skore._utils import repr_html

    template_dir = Path(repr_html.__file__).parent
    return Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)


def _create_method_tooltip_html(
    description: str, favorability_text: str | None = None
) -> str:
    """Create HTML for tooltip text content, optionally including favorability info."""
    # Escape HTML special characters in description for tooltip
    description_escaped = (
        description.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )

    # Build tooltip content
    tooltip_content = description_escaped
    if favorability_text:
        # Don't escape HTML in favorability_text if it contains HTML tags (like span for arrows)
        # Only escape if it doesn't already contain HTML
        if "<span" in favorability_text:
            favorability_escaped = favorability_text
        else:
            favorability_escaped = (
                favorability_text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;")
            )
        tooltip_content = f"{description_escaped}<br><br>{favorability_escaped}"

    return f'<span class="skore-help-tooltip-text">{tooltip_content}</span>'


class _RichHelpMixin(ABC):
    """Mixin for Rich-based help rendering."""

    def _create_help_tree(self) -> Tree:
        """Create the help tree for Rich rendering."""
        # Check if this is a report (has _ACCESSOR_CONFIG) or an accessor
        if hasattr(self, "_ACCESSOR_CONFIG"):
            # Report implementation
            tree = Tree(self.__class__.__name__)

            # Add accessor methods first
            for accessor_attr, config in self._ACCESSOR_CONFIG.items():
                accessor = getattr(self, accessor_attr)
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
            tree_title = (
                self._get_help_tree_title()
                if hasattr(self, "_get_help_tree_title")
                else self.__class__.__name__
            )
            tree = Tree(tree_title)

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

        return Panel(
            content,
            title=self._get_help_panel_title(),
            expand=False,
            border_style="orange1",
        )


class _HTMLHelpMixin(ABC):
    """Mixin for HTML-based help rendering with Shadow DOM isolation."""

    def _build_help_data(self) -> dict[str, Any]:
        """Build data structure for Jinja2 template rendering."""
        import uuid

        title = self._get_help_panel_title()
        title_clean = _strip_rich_markup(title)
        class_name = self.__class__.__name__

        data: dict[str, Any] = {
            "title": title_clean,
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
                        description_clean = _strip_rich_markup(description)
                        tooltip_html = _create_method_tooltip_html(description_clean)
                        doc_url = _get_documentation_url(class_name, None, attr_name)
                        data["attributes"].append(
                            {
                                "name": attr_name,
                                "tooltip_html": tooltip_html,
                                "doc_url": doc_url,
                            }
                        )
        else:
            # Accessor implementation
            tree_title = (
                self._get_help_tree_title()
                if hasattr(self, "_get_help_tree_title")
                else self.__class__.__name__
            )
            tree_title_clean = _strip_rich_markup(tree_title)
            data["tree_title"] = tree_title_clean
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
        """Build data structure for a single method."""
        displayed_name = obj._format_method_name(name, method)
        description = obj._get_method_description(method)
        displayed_name_clean = _strip_rich_markup(displayed_name)
        description_clean = _strip_rich_markup(description)

        # Split method name from parameters
        if "(" in displayed_name_clean:
            method_name_only, params_part = displayed_name_clean.split("(", 1)
            params_part = "(" + params_part
        else:
            method_name_only = displayed_name_clean
            params_part = ""

        # Get favorability text if applicable
        favorability_text = None
        if hasattr(obj, "_get_favorability_text"):
            favorability_text = obj._get_favorability_text(name)

        tooltip_html = _create_method_tooltip_html(description_clean, favorability_text)

        # Generate documentation URL
        doc_url = _get_documentation_url(class_name, accessor_path, name)

        return {
            "name_only": method_name_only,
            "params_part": params_part,
            "tooltip_html": tooltip_html,
            "doc_url": doc_url,
        }

    def _create_help_html(self) -> str:
        """Create the HTML representation of the help tree."""
        import uuid

        # Build data structure for template
        template_data = self._build_help_data()

        # Load template
        env = _get_jinja_env()
        template = env.get_template("help.html.j2")

        # Generate unique ID for this instance
        container_id = f"skore-help-{uuid.uuid4().hex[:8]}"

        # Render the template with all data
        # CSS and JS are included directly via {% include %} in the template
        shadow_dom_html = template.render(
            container_id=container_id,
            **template_data,
        )

        return shadow_dom_html


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

    def _is_interactive_environment(self) -> bool:
        """Check if we are in an interactive environment (IPython/Jupyter)."""
        try:
            from IPython import get_ipython

            ipython = get_ipython()
            return ipython is not None
        except ImportError:
            return False

    def help(self) -> None:
        """Display available methods using rich or HTML.

        In interactive environments (Jupyter/IPython), displays HTML representation
        with collapsible sections. Otherwise, displays Rich formatted output.
        """
        if self._is_interactive_environment():
            # Display HTML in interactive environments
            try:
                from IPython.display import HTML, display

                display(HTML(self._create_help_html()))
            except ImportError:
                # Fallback to Rich if IPython is not available
                from skore import console  # avoid circular import

                console.print(self._create_help_panel())
        else:
            # Display Rich in terminal
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


class _BaseReport(_HelpMixin):
    """Base class for all reports."""

    _ACCESSOR_CONFIG: dict[str, dict[str, str]]
    _X_train: ArrayLike | None
    _X_test: ArrayLike | None
    _y_train: ArrayLike | None
    _y_test: ArrayLike | None
    _cache: Cache
    estimator_: BaseEstimator

    def _get_help_panel_title(self) -> str:
        return ""

    def _get_attributes_for_help(self) -> list[str]:
        """Get the public attributes to display in help."""
        attributes = []
        xy_attributes = []

        for name in dir(self):
            # Skip private attributes, callables, and accessors
            if (
                name.startswith("_")
                or callable(getattr(self, name))
                or isinstance(getattr(self, name), _BaseAccessor)
            ):
                continue

            # Group X and y attributes separately
            value = getattr(self, name)
            if name.startswith(("X", "y")):
                if value is not None:  # Only include non-None X/y attributes
                    xy_attributes.append(name)
            else:
                attributes.append(name)

        # Sort X/y attributes to keep them grouped
        xy_attributes.sort()
        attributes.sort()

        # Return X/y attributes first, followed by other attributes
        return xy_attributes + attributes


ParentT = TypeVar("ParentT", bound="_BaseReport")


class _BaseAccessor(_HelpMixin, Generic[ParentT]):
    """Base class for all accessors."""

    def __init__(self, parent: ParentT) -> None:
        self._parent = parent

    @abstractmethod
    def _get_help_tree_title(self) -> str:
        """Get the title for the help tree."""
        pass

    def _get_help_panel_title(self) -> str:
        name = self.__class__.__name__.replace("_", "").replace("Accessor", "").lower()
        return f"Available {name} methods"

    def _get_X_y_and_data_source_hash(
        self,
        *,
        data_source: Literal["test", "train", "X_y"],
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
    ) -> tuple[ArrayLike, ArrayLike | None, int | None]:
        """Get the requested dataset and mention if we should hash before caching.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features) or None, default=None
            The input data.

        y : array-like of shape (n_samples,) or None, default=None
            The target data.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            The requested dataset.

        y : array-like of shape (n_samples,)
            The requested dataset.

        data_source_hash : int or None
            The hash of the data source. None when we are able to track the data, and
            thus relying on X_train, y_train, X_test, y_test.
        """
        is_cluster = is_clusterer(self._parent.estimator_)
        if data_source == "test":
            if not (X is None or y is None):
                raise ValueError("X and y must be None when data_source is test.")
            if self._parent._X_test is None or (
                not is_cluster and self._parent._y_test is None
            ):
                missing_data = "X_test" if is_cluster else "X_test and y_test"
                raise ValueError(
                    f"No {data_source} data (i.e. {missing_data}) were provided "
                    f"when creating the report. Please provide the {data_source} "
                    "data either when creating the report or by setting data_source "
                    "to 'X_y' and providing X and y."
                )
            return self._parent._X_test, self._parent._y_test, None
        elif data_source == "train":
            if not (X is None or y is None):
                raise ValueError("X and y must be None when data_source is train.")
            if self._parent._X_train is None or (
                not is_cluster and self._parent._y_train is None
            ):
                missing_data = "X_train" if is_cluster else "X_train and y_train"
                raise ValueError(
                    f"No {data_source} data (i.e. {missing_data}) were provided "
                    f"when creating the report. Please provide the {data_source} "
                    "data either when creating the report or by setting data_source "
                    "to 'X_y' and providing X and y."
                )
            return self._parent._X_train, self._parent._y_train, None
        elif data_source == "X_y":
            if X is None or (not is_cluster and y is None):
                missing_data = "X" if is_cluster else "X and y"
                raise ValueError(
                    f"{missing_data} must be provided when data_source is X_y."
                )
            return X, y, joblib.hash((X, y))
        else:
            raise ValueError(
                f"Invalid data source: {data_source}. Possible values are: "
                "test, train, X_y."
            )


def _get_cached_response_values(
    *,
    cache: Cache,
    estimator_hash: int,
    estimator: BaseEstimator,
    X: ArrayLike | None,
    response_method: str | list[str] | tuple[str, ...],
    pos_label: PositiveLabel | None = None,
    data_source: Literal["test", "train", "X_y"] = "test",
    data_source_hash: int | None = None,
) -> list[tuple[tuple[Any, ...], Any, bool]]:
    """Compute or load from local cache the response values.

    Be aware that the predictions will be loaded from the cache if present, but they
    will not be added to it. The reason is that we want to be able to run this function
    in parallel settings in a thread-safe manner. The update should be done outside of
    this function.

    Parameters
    ----------
    cache : dict
        The cache to use.

    estimator_hash : int
        A hash associated with the estimator such that we can retrieve the data from
        the cache.

    estimator : estimator object
        The estimator.

    X : {array-like, sparse matrix} of shape (n_samples, n_features) or None
        The data.

    response_method : str, list of str or tuple of str
        The response method.

    pos_label : int, float, bool or str, default=None
        The positive label.

    data_source : {"test", "train", "X_y"}, default="test"
        The data source to use.

        - "test" : use the test set provided when creating the report.
        - "train" : use the train set provided when creating the report.
        - "X_y" : use the provided `X` and `y` to compute the metric.

    data_source_hash : int or None
        The hash of the data source when `data_source` is "X_y".

    Returns
    -------
    list of tuples
        A list of tuples, each containing:

        - cache_key : tuple
            The cache key.

        - cache_value : Any
            The cache value. It corresponds to the predictions but also to the predict
            time when it has not been cached yet.

        - is_cached : bool
            Whether the cache value was loaded from the cache.
    """
    prediction_method = _check_response_method(estimator, response_method).__name__

    if data_source == "X_y" and data_source_hash is None:
        # Only trigger hash computation if it was not previously done.
        # If data_source_hash is not None, we internally computed ourself the hash
        # and it is trustful
        data_source_hash = joblib.hash(X)

    if prediction_method not in ("predict_proba", "decision_function"):
        # pos_label is only important in classification and with probabilities
        # and decision functions
        pos_label = None

    cache_key: tuple[Any, ...] = (
        estimator_hash,
        pos_label,
        prediction_method,
        data_source,
        data_source_hash,
    )

    if cache_key in cache:
        cached_predictions = cast(NDArray, cache[cache_key])
        return [(cache_key, cached_predictions, True)]

    with MeasureTime() as predict_time:
        predictions, _ = _get_response_values(
            estimator,
            X=X,
            response_method=prediction_method,
            pos_label=pos_label,
            return_response_method_used=False,
        )

    predict_time_cache_key: tuple[Any, ...] = (
        estimator_hash,
        data_source,
        data_source_hash,
        "predict_time",
    )

    return [
        (cache_key, predictions, False),
        (predict_time_cache_key, predict_time(), False),
    ]


class _BaseMetricsAccessor:
    _score_or_loss_info: dict[str, dict[str, str]] = {
        "fit_time": {"name": "Fit time (s)", "icon": "(↘︎)"},
        "predict_time": {"name": "Predict time (s)", "icon": "(↘︎)"},
        "accuracy": {"name": "Accuracy", "icon": "(↗︎)"},
        "precision": {"name": "Precision", "icon": "(↗︎)"},
        "recall": {"name": "Recall", "icon": "(↗︎)"},
        "brier_score": {"name": "Brier score", "icon": "(↘︎)"},
        "roc_auc": {"name": "ROC AUC", "icon": "(↗︎)"},
        "log_loss": {"name": "Log loss", "icon": "(↘︎)"},
        "r2": {"name": "R²", "icon": "(↗︎)"},
        "rmse": {"name": "RMSE", "icon": "(↘︎)"},
        "custom_metric": {"name": "Custom metric", "icon": ""},
        "report_metrics": {"name": "Report metrics", "icon": ""},
    }

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def _sort_methods_for_help(self, methods: list[tuple]) -> list[tuple]:
        """Override sort method for metrics-specific ordering.

        In short, we display the `summarize` first and then the `custom_metric`.
        """

        def _sort_key(method):
            name = method[0]
            if name == "custom_metric":
                priority = 1
            elif name == "summarize":
                priority = 2
            else:
                priority = 0
            return priority, name

        return sorted(methods, key=_sort_key)

    def _format_method_name(self, name: str, method: Any | None = None) -> str:
        """Override format method for metrics-specific naming."""
        # Get the signature from parent class
        if method is None:
            method = getattr(self, name, None)

        if method is None:
            method_name = f"{name}(...)"
        else:
            try:
                sig = inspect.signature(method)
                # Get parameter string, removing 'self' parameter
                params = []
                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue
                    params.append(param_name)

                params_str = ", ".join(params)
                method_name = f"{name}({params_str})"
            except (ValueError, TypeError):
                method_name = f"{name}(...)"

        return method_name.ljust(29)

    def _get_help_panel_title(self) -> str:
        return "[bold cyan]Available metrics methods[/bold cyan]"

    def _get_help_tree_title(self) -> str:
        return "[bold cyan]report.metrics[/bold cyan]"

    def _get_favorability_text(self, name: str) -> str | None:
        """Get favorability text for a method, or None if not applicable."""
        if name not in self._score_or_loss_info:
            return None
        icon = self._score_or_loss_info[name]["icon"]
        if icon == "(↗︎)":
            return 'higher is better <span class="skore-help-arrow">↗</span>'
        elif icon == "(↘︎)":
            return 'lower is better <span class="skore-help-arrow">↘</span>'
        return None
