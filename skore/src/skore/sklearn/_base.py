import inspect
import re
from abc import ABC, abstractmethod
from io import StringIO
from typing import Any, Generic, Literal, Optional, TypeVar, Union

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from rich.console import Console, Group
from rich.panel import Panel
from rich.tree import Tree
from sklearn.base import BaseEstimator
from sklearn.utils._response import _check_response_method, _get_response_values

from skore.externals._sklearn_compat import is_clusterer


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

    def _get_help_legend(self) -> Optional[str]:
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
        content: Union[Tree, Group]
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

    def _rich_repr(self, class_name: str, help_method_name: str) -> str:
        """Return a string representation using rich."""
        string_buffer = StringIO()
        console = Console(file=string_buffer, force_terminal=False)
        console.print(
            Panel(
                f"Get guidance using the {help_method_name} method",
                title=f"[cyan]{class_name}[/cyan]",
                border_style="orange1",
                expand=False,
            )
        )
        return string_buffer.getvalue()


class _BaseReport(_HelpMixin):
    """Base class for all reports."""

    _ACCESSOR_CONFIG: dict[str, dict[str, str]]
    _X_train: Optional[ArrayLike]
    _X_test: Optional[ArrayLike]
    _y_train: Optional[ArrayLike]
    _y_test: Optional[ArrayLike]
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
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
    ) -> tuple[ArrayLike, Optional[ArrayLike], Optional[int]]:
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
    cache: dict[tuple[Any, ...], ArrayLike],
    estimator_hash: int,
    estimator: BaseEstimator,
    X: Union[ArrayLike, None],
    response_method: Union[str, list[str], tuple[str, ...]],
    pos_label: Optional[Union[int, float, bool, str]] = None,
    data_source: Literal["test", "train", "X_y"] = "test",
    data_source_hash: Optional[int] = None,
) -> NDArray:
    """Compute or load from local cache the response values.

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
    array-like of shape (n_samples,) or (n_samples, n_outputs)
        The response values.
    """
    prediction_method = _check_response_method(estimator, response_method).__name__
    if prediction_method in ("predict_proba", "decision_function"):
        # pos_label is only important in classification and with probabilities
        # and decision functions
        cache_key: tuple[Any, ...] = (
            estimator_hash,
            pos_label,
            prediction_method,
            data_source,
        )
    else:
        cache_key = (estimator_hash, None, prediction_method, data_source)

    if data_source == "X_y":
        if data_source_hash is None:
            # Only trigger hash computation if it was not previously done.
            # If data_source_hash is not None, we internally computed ourself the hash
            # and it is trustful
            data_source_hash = joblib.hash(X)
        cache_key = cache_key + (data_source_hash,)

    if cache_key in cache:
        cached_predictions = cache[cache_key]
        assert isinstance(cached_predictions, np.ndarray)
        return cached_predictions

    predictions, _ = _get_response_values(
        estimator,
        X=X,
        response_method=prediction_method,
        pos_label=pos_label,
        return_response_method_used=False,
    )
    cache[cache_key] = predictions

    return predictions
