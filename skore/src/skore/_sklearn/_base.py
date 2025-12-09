import inspect
import re
from abc import ABC, abstractmethod
from io import StringIO
from typing import Any, Generic, Literal, TypeVar, cast

import joblib
from numpy.typing import ArrayLike, NDArray
from rich.console import Console, Group
from rich.panel import Panel
from rich.tree import Tree
from sklearn.base import BaseEstimator
from sklearn.utils._response import _check_response_method, _get_response_values

from skore._externals._sklearn_compat import is_clusterer
from skore._sklearn.types import PositiveLabel
from skore._utils._cache import Cache
from skore._utils._measure_time import MeasureTime


class _HelpMixin(ABC):
    """Mixin class providing help for the `help` method and the `__repr__` method."""

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

    def _format_method_name(self, name: str) -> str:
        """Format method name for display."""
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

    def _get_help_legend(self) -> str | None:
        """Get the help legend."""
        return None

    @abstractmethod
    def _create_help_tree(self) -> Tree:
        """Create the help tree."""

    @abstractmethod
    def _get_help_panel_title(self) -> str:
        """Get the help panel title."""

    def _create_help_panel(self) -> Panel:
        """Create the help panel."""
        content: Tree | Group
        if self._get_help_legend():
            content = Group(
                self._create_help_tree(),
                f"\n\nLegend:\n{self._get_help_legend()}",
            )
        else:
            content = self._create_help_tree()

        return Panel(
            content,
            title=self._get_help_panel_title(),
            expand=False,
            border_style="orange1",
        )

    def help(self) -> None:
        """Display available methods using rich."""
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

    def _get_help_legend(self) -> str:
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

    def _create_help_tree(self) -> Tree:
        """Create a rich Tree with the available tools and accessor methods."""
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
                displayed_name = accessor._format_method_name(name)
                description = accessor._get_method_description(method)
                branch.add(f".{displayed_name} - {description}")

            # Add sub-accessors after main methods
            for sub_attr, sub_obj in inspect.getmembers(accessor):
                if isinstance(sub_obj, _BaseAccessor) and not sub_attr.startswith("_"):
                    sub_branch = branch.add(f"[bold cyan].{sub_attr}[/bold cyan]")

                    # Add sub-accessor methods
                    sub_methods = sub_obj._get_methods_for_help()
                    sub_methods = sub_obj._sort_methods_for_help(sub_methods)

                    for name, method in sub_methods:
                        displayed_name = sub_obj._format_method_name(name)
                        description = sub_obj._get_method_description(method)
                        sub_branch.add(f".{displayed_name.ljust(25)} - {description}")

        # Add base methods
        base_methods = self._get_methods_for_help()
        base_methods = self._sort_methods_for_help(base_methods)

        for name, method in base_methods:
            description = self._get_method_description(method)
            tree.add(f".{name}(...)".ljust(34) + f" - {description}")

        # Add attributes section
        attributes = self._get_attributes_for_help()
        if attributes:
            attr_branch = tree.add("[bold cyan]Attributes[/bold cyan]")
            for attr_name in attributes:
                description = self._get_attribute_description(attr_name)
                attr_branch.add(f".{attr_name.ljust(29)} - {description}")

        return tree


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

    def _create_help_tree(self) -> Tree:
        """Create a rich Tree with the available methods."""
        tree = Tree(self._get_help_tree_title())

        methods = self._get_methods_for_help()
        methods = self._sort_methods_for_help(methods)

        for name, method in methods:
            displayed_name = self._format_method_name(name)
            description = self._get_method_description(method)
            tree.add(f".{displayed_name}".ljust(26) + f" - {description}")

        return tree

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

    def _format_method_name(self, name: str) -> str:
        """Override format method for metrics-specific naming."""
        method_name = f"{name}(...)"
        method_name = method_name.ljust(22)
        if name in self._score_or_loss_info and self._score_or_loss_info[name][
            "icon"
        ] in (
            "(↗︎)",
            "(↘︎)",
        ):
            if self._score_or_loss_info[name]["icon"] == "(↗︎)":
                method_name += f"[cyan]{self._score_or_loss_info[name]['icon']}[/cyan]"
                return method_name.ljust(43)
            else:  # (↘︎)
                method_name += (
                    f"[orange1]{self._score_or_loss_info[name]['icon']}[/orange1]"
                )
                return method_name.ljust(49)
        else:
            return method_name.ljust(29)

    def _get_help_panel_title(self) -> str:
        return "[bold cyan]Available metrics methods[/bold cyan]"

    def _get_help_tree_title(self) -> str:
        return "[bold cyan]report.metrics[/bold cyan]"

    def _get_help_legend(self) -> str:
        return (
            "[cyan](↗︎)[/cyan] higher is better [orange1](↘︎)[/orange1] lower is better"
        )
