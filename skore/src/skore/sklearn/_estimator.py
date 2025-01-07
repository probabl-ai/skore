import inspect
import time
from functools import partial
from io import StringIO
from itertools import product

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.tree import Tree
from sklearn import metrics
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics._scorer import _BaseScorer
from sklearn.pipeline import Pipeline
from sklearn.utils._response import _check_response_method, _get_response_values
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from skore.externals._sklearn_compat import is_clusterer
from skore.sklearn._plot import (
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from skore.sklearn.find_ml_task import _find_ml_task
from skore.utils._accessor import _check_supported_ml_task, register_accessor


def _check_supported_estimator(supported_estimators):
    def check(accessor):
        estimator = accessor._parent.estimator
        if isinstance(estimator, Pipeline):
            estimator = estimator.steps[-1][1]
        supported_estimator = isinstance(estimator, supported_estimators)

        if not supported_estimator:
            raise AttributeError(
                f"The {estimator.__class__.__name__} estimator is not supported "
                "by the function called."
            )

        return True

    return check


class _HelpMixin:
    """Mixin class providing help for the `help` method and the `__repr__` method."""

    def _get_methods_for_help(self):
        """Get the methods to display in help."""
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        filtered_methods = []
        for name, method in methods:
            is_private_method = name.startswith("_")
            # we cannot use `isinstance(method, classmethod)` because it is already
            # already transformed by the decorator `@classmethod`.
            is_class_method = inspect.ismethod(method) and method.__self__ is type(self)
            is_help_method = name == "help"
            if not (is_private_method or is_class_method or is_help_method):
                filtered_methods.append((name, method))
        return filtered_methods

    def _sort_methods_for_help(self, methods):
        """Sort methods for help display."""
        return sorted(methods)

    def _format_method_name(self, name):
        """Format method name for display."""
        return f"{name}(...)"

    def _get_method_description(self, method):
        """Get the description for a method."""
        return (
            method.__doc__.split("\n")[0]
            if method.__doc__
            else "No description available"
        )

    def _create_help_panel(self):
        """Create the help panel."""
        return Panel(
            self._create_help_tree(),
            title=self._get_help_panel_title(),
            expand=False,
            border_style="orange1",
        )

    def help(self):
        """Display available methods using rich."""
        from skore import console  # avoid circular import

        console.print(self._create_help_panel())

    def __repr__(self):
        """Return a string representation using rich."""
        console = Console(file=StringIO(), force_terminal=False)
        console.print(self._create_help_panel())
        return console.file.getvalue()


########################################################################################
# EstimatorReport
########################################################################################


class EstimatorReport(_HelpMixin):
    """Report for a fitted estimator.

    This class provides a set of tools to quickly validate and inspect a scikit-learn
    compatible estimator.

    Parameters
    ----------
    estimator : estimator object
        Estimator to make report from.

    fit : {"auto", True, False}, default="auto"
        Whether to fit the estimator on the training data. If "auto", the estimator
        is fitted only if the training data is provided.

    X_train : {array-like, sparse matrix} of shape (n_samples, n_features) or \
            None
        Training data.

    y_train : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        Training target.

    X_test : {array-like, sparse matrix} of shape (n_samples, n_features) or None
        Testing data. It should have the same structure as the training data.

    y_test : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        Testing target.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> estimator = LogisticRegression().fit(X_train, y_train)
    >>> from skore import EstimatorReport
    >>> report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    """

    _ACCESSOR_CONFIG = {
        "metrics": {"icon": "📏", "name": "metrics"},
        # Add other accessors as they're implemented
        # "inspection": {"icon": "🔍", "name": "inspection"},
        # "linting": {"icon": "✔️", "name": "linting"},
    }

    @staticmethod
    def _fit_estimator(estimator, X_train, y_train):
        if X_train is None or (y_train is None and not is_clusterer(estimator)):
            raise ValueError(
                "The training data is required to fit the estimator. "
                "Please provide both X_train and y_train."
            )
        return clone(estimator).fit(X_train, y_train)

    def __init__(
        self,
        estimator,
        *,
        fit="auto",
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
    ):
        if fit == "auto":
            try:
                check_is_fitted(estimator)
                self._estimator = estimator
            except NotFittedError:
                self._estimator = self._fit_estimator(estimator, X_train, y_train)
        elif fit is True:
            self._estimator = self._fit_estimator(estimator, X_train, y_train)
        else:  # fit is False
            self._estimator = estimator

        # private storage to be able to invalidate the cache when the user alters
        # those attributes
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test

        self._initialize_state()

    def _initialize_state(self):
        """Initialize/reset the random number generator, hash, and cache."""
        self._rng = np.random.default_rng(time.time_ns())
        self._hash = self._rng.integers(
            low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max
        )
        self._cache = {}
        self._ml_task = _find_ml_task(self._y_test, estimator=self._estimator)

    # NOTE:
    # For the moment, we do not allow to alter the estimator and the training data.
    # For the validation set, we allow it and we invalidate the cache.

    def clean_cache(self):
        """Clean the cache."""
        self._cache = {}

    def cache_predictions(self, response_methods="auto", n_jobs=None):
        """Force caching of estimator's predictions.

        Parameters
        ----------
        response_methods : "auto" or list of str, default="auto"
            The response methods to precompute. If "auto", the response methods are
            inferred from the ml task: for classification we compute the response of
            the `predict_proba`, `decision_function` and `predict` methods; for
            regression we compute the response of the `predict` method.

        n_jobs : int or None, default=None
            The number of jobs to run in parallel. None means 1 unless in a
            joblib.parallel_backend context. -1 means using all processors.
        """
        if self._ml_task in ("binary-classification", "multiclass-classification"):
            if response_methods == "auto":
                response_methods = ("predict",)
                if hasattr(self._estimator, "predict_proba"):
                    response_methods = ("predict_proba",)
                if hasattr(self._estimator, "decision_function"):
                    response_methods = ("decision_function",)
            pos_labels = self._estimator.classes_
        else:
            if response_methods == "auto":
                response_methods = ("predict",)
            pos_labels = [None]

        data_sources = ("test",)
        Xs = (self._X_test,)
        if self._X_train is not None:
            data_sources = ("train",)
            Xs = (self._X_train,)

        parallel = joblib.Parallel(n_jobs=n_jobs, return_as="generator_unordered")
        generator = parallel(
            joblib.delayed(self._get_cached_response_values)(
                estimator_hash=self._hash,
                estimator=self._estimator,
                X=X,
                response_method=response_method,
                pos_label=pos_label,
                data_source=data_source,
            )
            for response_method, pos_label, data_source, X in product(
                response_methods, pos_labels, data_sources, Xs
            )
        )
        # trigger the computation
        list(
            track(
                generator,
                total=len(response_methods) * len(pos_labels) * len(data_sources),
                description="Caching predictions",
            )
        )

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        raise AttributeError(
            "The estimator attribute is immutable. "
            "Call the constructor of {self.__class__.__name__} to create a new report."
        )

    @property
    def X_train(self):
        return self._X_train

    @X_train.setter
    def X_train(self, value):
        raise AttributeError(
            "The X_train attribute is immutable. "
            "Please use the `from_unfitted_estimator` method to create a new report."
        )

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        raise AttributeError(
            "The y_train attribute is immutable. "
            "Please use the `from_unfitted_estimator` method to create a new report."
        )

    @property
    def X_test(self):
        return self._X_test

    @X_test.setter
    def X_test(self, value):
        self._X_test = value
        self._initialize_state()

    @property
    def y_test(self):
        return self._y_test

    @y_test.setter
    def y_test(self, value):
        self._y_test = value
        self._initialize_state()

    @property
    def estimator_name(self):
        if isinstance(self._estimator, Pipeline):
            name = self._estimator[-1].__class__.__name__
        else:
            name = self._estimator.__class__.__name__
        return name

    def _get_cached_response_values(
        self,
        *,
        estimator_hash,
        estimator,
        X,
        response_method,
        pos_label=None,
        data_source="test",
        data_source_hash=None,
    ):
        """Compute or load from local cache the response values.

        Parameters
        ----------
        estimator_hash : int
            A hash associated with the estimator such that we can retrieve the data from
            the cache.

        estimator : estimator object
            The estimator.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data.

        response_method : str
            The response method.

        pos_label : str, default=None
            The positive label.

        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.
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
            cache_key = (estimator_hash, pos_label, prediction_method, data_source)
        else:
            cache_key = (estimator_hash, prediction_method, data_source)

        if data_source == "X_y":
            data_source_hash = joblib.hash(X)
            cache_key += (data_source_hash,)

        if cache_key in self._cache:
            return self._cache[cache_key]

        predictions, _ = _get_response_values(
            estimator,
            X=X,
            response_method=prediction_method,
            pos_label=pos_label,
            return_response_method_used=False,
        )
        self._cache[cache_key] = predictions

        return predictions

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def _get_help_panel_title(self):
        return (
            f"[bold cyan]📓 Tools to diagnose {self.estimator_name} "
            "estimator[/bold cyan]"
        )

    def _create_help_tree(self):
        """Create a rich Tree with the available tools and accessor methods."""
        tree = Tree("reporter")

        # Add accessor methods first
        for accessor_attr, config in self._ACCESSOR_CONFIG.items():
            accessor = getattr(self, accessor_attr)
            branch = tree.add(
                f"[bold cyan].{config['name']} {config['icon']}[/bold cyan]"
            )

            # Add main accessor methods first
            methods = accessor._get_methods_for_help()
            methods = accessor._sort_methods_for_help(methods)

            for name, method in methods:
                displayed_name = accessor._format_method_name(name)
                description = accessor._get_method_description(method)
                branch.add(f".{displayed_name} - {description}")

            # Add sub-accessors after main methods
            for sub_attr, sub_obj in inspect.getmembers(accessor):
                if isinstance(sub_obj, _BaseAccessor) and not sub_attr.startswith("_"):
                    sub_branch = branch.add(
                        f"[bold cyan].{sub_attr} {sub_obj._icon}[/bold cyan]"
                    )

                    # Add sub-accessor methods
                    sub_methods = sub_obj._get_methods_for_help()
                    sub_methods = sub_obj._sort_methods_for_help(sub_methods)

                    for name, method in sub_methods:
                        displayed_name = sub_obj._format_method_name(name)
                        description = sub_obj._get_method_description(method)
                        sub_branch.add(f".{displayed_name.ljust(25)} - {description}")

        # Add base methods last
        base_methods = self._get_methods_for_help()
        base_methods = self._sort_methods_for_help(base_methods)

        for name, method in base_methods:
            description = self._get_method_description(method)
            tree.add(f".{name}(...)".ljust(34) + f" - {description}")

        return tree


########################################################################################
# Base class for the accessors
########################################################################################


class _BaseAccessor(_HelpMixin):
    """Base class for all accessors."""

    def __init__(self, parent, icon):
        self._parent = parent
        self._icon = icon

    def _get_help_panel_title(self):
        name = self.__class__.__name__.replace("_", "").replace("Accessor", "").lower()
        return f"{self._icon} Available {name} methods"

    def _create_help_tree(self):
        """Create a rich Tree with the available methods."""
        tree = Tree(self._get_help_tree_title())

        methods = self._get_methods_for_help()
        methods = self._sort_methods_for_help(methods)

        for name, method in methods:
            displayed_name = self._format_method_name(name)
            description = self._get_method_description(method)
            tree.add(f".{displayed_name}".ljust(26) + f" - {description}")

        return tree

    def _get_X_y_and_data_source_hash(self, *, data_source, X=None, y=None):
        """Get the requested dataset and mention if we should hash before caching.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.
            - "X_y" : use the provided `X` and `y` to compute the metric.

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
        if data_source == "test":
            if not (X is None or y is None):
                raise ValueError("X and y must be None when data_source is test.")
            return self._parent._X_test, self._parent._y_test, None
        elif data_source == "train":
            if not (X is None or y is None):
                raise ValueError("X and y must be None when data_source is train.")
            is_cluster = is_clusterer(self._parent.estimator)
            if self._parent._X_train is None or (
                not is_cluster and self._parent._y_train is None
            ):
                missing_data = "X_train" if is_cluster else "X_train and y_train"
                raise ValueError(
                    f"No training data (i.e. {missing_data}) were provided "
                    "when creating the reporter. Please provide the training data."
                )
            return self._parent._X_train, self._parent._y_train, None
        elif data_source == "X_y":
            is_cluster = is_clusterer(self._parent.estimator)
            if X is None or (not is_cluster and y is None):
                missing_data = "X" if is_cluster else "X and y"
                raise ValueError(f"{missing_data} must be provided.")
            return X, y, joblib.hash((X, y))
        else:
            raise ValueError(
                f"Invalid data source: {data_source}. Possible values are: "
                "test, train, X_y."
            )


########################################################################################
# Plotting accessor for metrics
########################################################################################


class _PlotMetricsAccessor(_BaseAccessor):
    def __init__(self, parent):
        # Note: parent here will be the MetricsAccessor instance
        super().__init__(parent._parent, icon="🎨")
        self._metrics_parent = parent

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def roc(self, *, data_source="test", X=None, y=None, pos_label=None, ax=None):
        """Plot the ROC curve.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the reporter.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the reporter.

        pos_label : str, default=None
            The positive class.

        ax : matplotlib.axes.Axes, default=None
            The axes to plot on.

        Returns
        -------
        RocCurveDisplay
            The ROC curve display.
        """
        prediction_method = ["predict_proba", "decision_function"]
        X, y, data_source_hash = self._get_X_y_and_data_source_hash(
            data_source=data_source, X=X, y=y
        )

        cache_key = (self._parent._hash, RocCurveDisplay.__name__, pos_label)

        if data_source_hash:  # data_source == "X_y"
            cache_key += (data_source_hash,)
        else:
            cache_key += (data_source,)

        if cache_key in self._parent._cache:
            display = self._parent._cache[cache_key]
            display.plot(
                ax=ax,
                estimator_name=self._parent.estimator_name,
                plot_chance_level=True,
                despine=True,
            )
        else:
            y_pred = self._parent._get_cached_response_values(
                estimator_hash=self._parent._hash,
                estimator=self._parent.estimator,
                X=X,
                response_method=prediction_method,
                pos_label=pos_label,
                data_source=data_source,
                data_source_hash=data_source_hash,
            )

            display = RocCurveDisplay._from_predictions(
                y,
                y_pred,
                estimator=self._parent.estimator,
                ml_task=self._parent._ml_task,
                data_source=data_source,
                pos_label=pos_label,
                ax=ax,
                estimator_name=self._parent.estimator_name,
                plot_chance_level=True,
                despine=True,
            )
            self._parent._cache[cache_key] = display

        return display

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def precision_recall(
        self,
        *,
        data_source="test",
        X=None,
        y=None,
        pos_label=None,
        ax=None,
    ):
        """Plot the precision-recall curve.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the reporter.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the reporter.

        pos_label : str, default=None
            The positive class.

        ax : matplotlib.axes.Axes, default=None
            The axes to plot on.

        Returns
        -------
        PrecisionRecallCurveDisplay
            The precision-recall curve display.
        """
        prediction_method = ["predict_proba", "decision_function"]
        X, y, data_source_hash = self._get_X_y_and_data_source_hash(
            data_source=data_source, X=X, y=y
        )

        cache_key = (
            self._parent._hash,
            PrecisionRecallCurveDisplay.__name__,
            pos_label,
        )
        if data_source_hash:  # data_source == "X_y"
            cache_key += (data_source_hash,)
        else:
            cache_key += (data_source,)

        if cache_key in self._parent._cache:
            display = self._parent._cache[cache_key]
            display.plot(
                ax=ax,
                estimator_name=self._parent.estimator_name,
                plot_chance_level=False,
                despine=True,
            )
        else:
            y_pred = self._parent._get_cached_response_values(
                estimator_hash=self._parent._hash,
                estimator=self._parent.estimator,
                X=X,
                response_method=prediction_method,
                pos_label=pos_label,
                data_source=data_source,
                data_source_hash=data_source_hash,
            )

            display = PrecisionRecallCurveDisplay._from_predictions(
                y,
                y_pred,
                estimator=self._parent.estimator,
                estimator_name=self._parent.estimator_name,
                ml_task=self._parent._ml_task,
                data_source=data_source,
                pos_label=pos_label,
                ax=ax,
                plot_chance_level=False,
                despine=True,
            )
            self._parent._cache[cache_key] = display

        return display

    @available_if(_check_supported_ml_task(supported_ml_tasks=["regression"]))
    def prediction_error(
        self,
        *,
        data_source="test",
        X=None,
        y=None,
        ax=None,
        kind="residual_vs_predicted",
        subsample=1_000,
    ):
        """Plot the prediction error of a regression model.

        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the reporter.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the reporter.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        kind : {"actual_vs_predicted", "residual_vs_predicted"}, \
                default="residual_vs_predicted"
            The type of plot to draw:

            - "actual_vs_predicted" draws the observed values (y-axis) vs.
              the predicted values (x-axis).
            - "residual_vs_predicted" draws the residuals, i.e. difference
              between observed and predicted values, (y-axis) vs. the predicted
              values (x-axis).

        subsample : float, int or None, default=1_000
            Sampling the samples to be shown on the scatter plot. If `float`,
            it should be between 0 and 1 and represents the proportion of the
            original dataset. If `int`, it represents the number of samples
            display on the scatter plot. If `None`, no subsampling will be
            applied. by default, 1,000 samples or less will be displayed.

        Returns
        -------
        PredictionErrorDisplay
            The prediction error display.
        """
        X, y, data_source_hash = self._get_X_y_and_data_source_hash(
            data_source=data_source, X=X, y=y
        )

        cache_key = (
            self._parent._hash,
            PredictionErrorDisplay.__name__,
            kind,
            subsample,
        )
        if data_source_hash:  # data_source == "X_y"
            cache_key += (data_source_hash,)
        else:
            cache_key += (data_source,)

        if cache_key in self._parent._cache:
            display = self._parent._cache[cache_key]
            display.plot(
                ax=ax,
                kind=kind,
            )
        else:
            y_pred = self._parent._get_cached_response_values(
                estimator_hash=self._parent._hash,
                estimator=self._parent.estimator,
                X=X,
                response_method="predict",
                data_source=data_source,
                data_source_hash=data_source_hash,
            )

            display = PredictionErrorDisplay._from_predictions(
                y,
                y_pred,
                estimator=self._parent.estimator,
                estimator_name=self._parent.estimator_name,
                ml_task=self._parent._ml_task,
                data_source=data_source,
                ax=ax,
                kind=kind,
                subsample=subsample,
            )
            self._parent._cache[cache_key] = display

        return display

    def _get_help_panel_title(self):
        return f"[bold cyan]{self._icon} Available plot methods[/bold cyan]"

    def _get_help_tree_title(self):
        return "[bold cyan]reporter.metrics.plot[/bold cyan]"


###############################################################################
# Metrics accessor
###############################################################################


@register_accessor("metrics", EstimatorReport)
class _MetricsAccessor(_BaseAccessor):
    _SCORE_OR_LOSS_ICONS = {
        "accuracy": "(↗︎)",
        "precision": "(↗︎)",
        "recall": "(↗︎)",
        "brier_score": "(↘︎)",
        "roc_auc": "(↗︎)",
        "log_loss": "(↘︎)",
        "r2": "(↗︎)",
        "rmse": "(↘︎)",
        "report_metrics": "",
        "custom_metric": "",
    }

    def __init__(self, parent):
        super().__init__(parent, icon="📏")
        # Create plot sub-accessor
        self.plot = _PlotMetricsAccessor(self)

    # TODO: should build on the `add_scorers` function
    def report_metrics(
        self,
        *,
        data_source="test",
        X=None,
        y=None,
        scoring=None,
        pos_label=1,
        scoring_kwargs=None,
    ):
        """Report a set of metrics for our estimator.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the reporter.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the reporter.

        scoring : list of str, callable, or scorer, default=None
            The metrics to report. You can get the possible list of string by calling
            `reporter.metrics.help()`. When passing a callable, it should take as
            arguments `y_true`, `y_pred` as the two first arguments. Additional
            arguments can be passed as keyword arguments and will be forwarded with
            `scoring_kwargs`. If the callable API is too restrictive (e.g. need to pass
            same parameter name with different values), you can use scikit-learn scorers
            as provided by :func:`sklearn.metrics.make_scorer`.

        pos_label : int, default=1
            The positive class.

        scoring_kwargs : dict, default=None
            The keyword arguments to pass to the scoring functions.

        Returns
        -------
        pd.DataFrame
            The statistics for the metrics.
        """
        if scoring is None:
            # Equivalent to _get_scorers_to_add
            if self._parent._ml_task == "binary-classification":
                scoring = ["precision", "recall", "roc_auc"]
                if hasattr(self._parent._estimator, "predict_proba"):
                    scoring.append("brier_score")
            elif self._parent._ml_task == "multiclass-classification":
                scoring = ["precision", "recall"]
                if hasattr(self._parent._estimator, "predict_proba"):
                    scoring += ["roc_auc", "log_loss"]
            else:
                scoring = ["r2", "rmse"]

        scores = []

        for metric in scoring:
            # NOTE: we have to check specifically for `_BaseScorer` first because this
            # is also a callable but it has a special private API that we can leverage
            if isinstance(metric, _BaseScorer):
                # scorers have the advantage to have scoped defined kwargs
                metric_fn = partial(
                    self.custom_metric,
                    metric_function=metric._score_func,
                    response_method=metric._response_method,
                )
                # forward the additional parameters specific to the scorer
                metrics_kwargs = {**metric._kwargs}
            elif isinstance(metric, str) or callable(metric):
                if isinstance(metric, str):
                    metric_fn = getattr(self, metric)
                    metrics_kwargs = {}
                else:
                    metric_fn = partial(self.custom_metric, metric_function=metric)
                    if scoring_kwargs is None:
                        metrics_kwargs = {}
                    else:
                        # check if we should pass any parameters specific to the metric
                        # callable
                        metric_callable_params = inspect.signature(metric).parameters
                        metrics_kwargs = {
                            param: scoring_kwargs[param]
                            for param in metric_callable_params
                            if param in scoring_kwargs
                        }
                metrics_params = inspect.signature(metric_fn).parameters
                if scoring_kwargs is not None:
                    for param in metrics_params:
                        if param in scoring_kwargs:
                            metrics_kwargs[param] = scoring_kwargs[param]
                if "pos_label" in metrics_params:
                    metrics_kwargs["pos_label"] = pos_label
            else:
                raise ValueError(
                    f"Invalid type of metric: {type(metric)} for {metric!r}"
                )

            scores.append(
                metric_fn(data_source=data_source, X=X, y=y, **metrics_kwargs)
            )

        has_multilevel = any(
            isinstance(score, pd.DataFrame) and isinstance(score.columns, pd.MultiIndex)
            for score in scores
        )

        if has_multilevel:
            # Convert single-level dataframes to multi-level
            for i, score in enumerate(scores):
                if hasattr(score, "columns") and not isinstance(
                    score.columns, pd.MultiIndex
                ):
                    name_index = (
                        ["Metric", "Output"]
                        if self._parent._ml_task == "regression"
                        else ["Metric", "Class label"]
                    )
                    scores[i].columns = pd.MultiIndex.from_tuples(
                        [(col, "") for col in score.columns],
                        names=name_index,
                    )

        return pd.concat(scores, axis=1)

    def _compute_metric_scores(
        self,
        metric_fn,
        X,
        y_true,
        *,
        response_method,
        pos_label=None,
        metric_name=None,
        data_source="test",
        data_source_hash=None,
        **metric_kwargs,
    ):
        y_pred = self._parent._get_cached_response_values(
            estimator_hash=self._parent._hash,
            estimator=self._parent.estimator,
            X=X,
            response_method=response_method,
            pos_label=pos_label,
            data_source=data_source,
            data_source_hash=data_source_hash,
        )
        cache_key = (self._parent._hash, metric_fn.__name__, data_source)
        if data_source_hash:
            cache_key += (data_source_hash,)

        metric_params = inspect.signature(metric_fn).parameters
        if "pos_label" in metric_params:
            cache_key += (pos_label,)
        if metric_kwargs != {}:
            # we need to enforce the order of the parameter for a specific metric
            # to make sure that we hit the cache in a consistent way
            ordered_metric_kwargs = sorted(metric_kwargs.keys())
            cache_key += tuple(
                (
                    joblib.hash(metric_kwargs[key])
                    if isinstance(metric_kwargs[key], np.ndarray)
                    else metric_kwargs[key]
                )
                for key in ordered_metric_kwargs
            )

        if cache_key in self._parent._cache:
            score = self._parent._cache[cache_key]
        else:
            metric_params = inspect.signature(metric_fn).parameters
            kwargs = {**metric_kwargs}
            if "pos_label" in metric_params:
                kwargs.update(pos_label=pos_label)

            score = metric_fn(y_true, y_pred, **kwargs)
            self._parent._cache[cache_key] = score

        score = np.array([score]) if not isinstance(score, np.ndarray) else score
        metric_name = metric_name or metric_fn.__name__

        if self._parent._ml_task in [
            "binary-classification",
            "multiclass-classification",
        ]:
            if len(score) == 1:
                columns = pd.Index([metric_name], name="Metric")
            else:
                classes = self._parent._estimator.classes_
                columns = pd.MultiIndex.from_arrays(
                    [[metric_name] * len(classes), classes],
                    names=["Metric", "Class label"],
                )
                score = score.reshape(1, -1)
        elif self._parent._ml_task == "regression":
            if len(score) == 1:
                columns = pd.Index([metric_name], name="Metric")
            else:
                columns = pd.MultiIndex.from_arrays(
                    [
                        [metric_name] * len(score),
                        [f"#{i}" for i in range(len(score))],
                    ],
                    names=["Metric", "Output"],
                )
                score = score.reshape(1, -1)
        else:
            # FIXME: clusterer would fall here.
            columns = None
        return pd.DataFrame(score, columns=columns, index=[self._parent.estimator_name])

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def accuracy(self, *, data_source="test", X=None, y=None):
        """Compute the accuracy score.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the reporter.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the reporter.

        Returns
        -------
        pd.DataFrame
            The accuracy score.
        """
        X, y, data_source_hash = self._get_X_y_and_data_source_hash(
            data_source=data_source, X=X, y=y
        )

        return self._compute_metric_scores(
            metrics.accuracy_score,
            X=X,
            y_true=y,
            response_method="predict",
            metric_name=f"Accuracy {self._SCORE_OR_LOSS_ICONS['accuracy']}",
            data_source_hash=data_source_hash,
            data_source=data_source,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def precision(
        self, *, data_source="test", X=None, y=None, average="auto", pos_label=1
    ):
        """Compute the precision score.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the reporter.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the reporter.

        average : {"auto", "macro", "micro", "weighted", "samples"} or None, \
                default="auto"
            The average to compute the precision score. By default, the average is
            "binary" for binary classification and "weighted" for multiclass
            classification.

        pos_label : int, default=1
            The positive class.

        Returns
        -------
        pd.DataFrame
            The precision score.
        """
        X, y, data_source_hash = self._get_X_y_and_data_source_hash(
            data_source=data_source, X=X, y=y
        )

        if average == "auto":
            if self._parent._ml_task == "binary-classification":
                average = "binary"
            else:
                average = "weighted"

        if average != "binary":
            # overwrite `pos_label` because it will be ignored
            pos_label = None

        return self._compute_metric_scores(
            metrics.precision_score,
            X=X,
            y_true=y,
            response_method="predict",
            pos_label=pos_label,
            metric_name=f"Precision {self._SCORE_OR_LOSS_ICONS['precision']}",
            average=average,
            data_source_hash=data_source_hash,
            data_source=data_source,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def recall(
        self, *, data_source="test", X=None, y=None, average="auto", pos_label=1
    ):
        """Compute the recall score.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the reporter.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the reporter.

        average : {"auto", "macro", "micro", "weighted", "samples"} or None, \
                default="auto"
            The average to compute the recall score. By default, the average is
            "binary" for binary classification and "weighted" for multiclass
            classification.

        pos_label : int, default=1
            The positive class.

        Returns
        -------
        pd.DataFrame
            The recall score.
        """
        X, y, data_source_hash = self._get_X_y_and_data_source_hash(
            data_source=data_source, X=X, y=y
        )

        if average == "auto":
            if self._parent._ml_task == "binary-classification":
                average = "binary"
            else:
                average = "weighted"

        if average != "binary":
            # overwrite `pos_label` because it will be ignored
            pos_label = None

        return self._compute_metric_scores(
            metrics.recall_score,
            X=self._parent._X_test,
            y_true=self._parent._y_test,
            response_method="predict",
            pos_label=pos_label,
            metric_name=f"Recall {self._SCORE_OR_LOSS_ICONS['recall']}",
            average=average,
            data_source_hash=data_source_hash,
            data_source=data_source,
        )

    @available_if(
        _check_supported_ml_task(supported_ml_tasks=["binary-classification"])
    )
    def brier_score(self, *, data_source="test", X=None, y=None, pos_label=1):
        """Compute the Brier score.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the reporter.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the reporter.

        pos_label : int, default=1
            The positive class.

        Returns
        -------
        pd.DataFrame
            The Brier score.
        """
        X, y, data_source_hash = self._get_X_y_and_data_source_hash(
            data_source=data_source, X=X, y=y
        )

        return self._compute_metric_scores(
            metrics.brier_score_loss,
            X=X,
            y_true=y,
            response_method="predict_proba",
            metric_name=f"Brier score {self._SCORE_OR_LOSS_ICONS['brier_score']}",
            pos_label=pos_label,
            data_source_hash=data_source_hash,
            data_source=data_source,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def roc_auc(
        self, *, data_source="test", X=None, y=None, average="auto", multi_class="ovr"
    ):
        """Compute the ROC AUC score.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the reporter.
            - "train" : use the train set provided when creating the reporter.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the reporter.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the reporter.

        average : {"auto", "macro", "micro", "weighted", "samples"}, \
                default="auto"
            The average to compute the ROC AUC score. By default, the average is "macro"
            for binary classification with probability predictions and "weighted" for
            multiclass classification with 1-vs-rest predictions.

        multi_class : {"raise", "ovr", "ovo", "auto"}, default="ovr"
            The multi-class strategy to use.

            - "raise" : Raise an error if the data is multiclass.
            - "ovr": Stands for One-vs-rest. Computes the AUC of each class against the
              rest. This treats the multiclass case in the same way as the multilabel
              case. Sensitive to class imbalance even when ``average == 'macro'``,
              because class imbalance affects the composition of each of the 'rest'
              groupings.
            - "ovo": Stands for One-vs-one. Computes the average AUC of all possible
              pairwise combinations of classes. Insensitive to class imbalance when
              ``average == 'macro'``.

        Returns
        -------
        pd.DataFrame
            The ROC AUC score.
        """
        X, y, data_source_hash = self._get_X_y_and_data_source_hash(
            data_source=data_source, X=X, y=y
        )

        if average == "auto":
            if self._parent._ml_task == "binary-classification":
                average = "macro"
            else:
                average = "weighted"

        return self._compute_metric_scores(
            metrics.roc_auc_score,
            X=X,
            y_true=y,
            response_method=["predict_proba", "decision_function"],
            metric_name=f"ROC AUC {self._SCORE_OR_LOSS_ICONS['roc_auc']}",
            average=average,
            multi_class=multi_class,
            data_source_hash=data_source_hash,
            data_source=data_source,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def log_loss(self, *, data_source="test", X=None, y=None):
        """Compute the log loss.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the reporter.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the reporter.

        Returns
        -------
        pd.DataFrame
            The log-loss.
        """
        X, y, data_source_hash = self._get_X_y_and_data_source_hash(
            data_source=data_source, X=X, y=y
        )

        return self._compute_metric_scores(
            metrics.log_loss,
            X=X,
            y_true=y,
            response_method="predict_proba",
            metric_name=f"Log loss {self._SCORE_OR_LOSS_ICONS['log_loss']}",
            data_source_hash=data_source_hash,
            data_source=data_source,
        )

    @available_if(_check_supported_ml_task(supported_ml_tasks=["regression"]))
    def r2(self, *, data_source="test", X=None, y=None, multioutput="uniform_average"):
        """Compute the R² score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the reporter.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the reporter.

        multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
                (n_outputs,), default='uniform_average'
            Defines aggregating of multiple output values.
            Array-like value defines weights used to average errors.

            'raw_values' :
                Returns a full set of errors in case of multioutput input.

            'uniform_average' :
                Errors of all outputs are averaged with uniform weight.

        Returns
        -------
        pd.DataFrame
            The R² score.
        """
        X, y, data_source_hash = self._get_X_y_and_data_source_hash(
            data_source=data_source, X=X, y=y
        )

        return self._compute_metric_scores(
            metrics.r2_score,
            X=X,
            y_true=y,
            response_method="predict",
            metric_name=f"R² {self._SCORE_OR_LOSS_ICONS['r2']}",
            multioutput=multioutput,
            data_source_hash=data_source_hash,
            data_source=data_source,
        )

    @available_if(_check_supported_ml_task(supported_ml_tasks=["regression"]))
    def rmse(
        self, *, data_source="test", X=None, y=None, multioutput="uniform_average"
    ):
        """Compute the root mean squared error.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the reporter.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the reporter.

        multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
                (n_outputs,), default='uniform_average'
            Defines aggregating of multiple output values.
            Array-like value defines weights used to average errors.

            'raw_values' :
                Returns a full set of errors in case of multioutput input.

            'uniform_average' :
                Errors of all outputs are averaged with uniform weight.

        Returns
        -------
        pd.DataFrame
            The root mean squared error.
        """
        X, y, data_source_hash = self._get_X_y_and_data_source_hash(
            data_source=data_source, X=X, y=y
        )

        return self._compute_metric_scores(
            metrics.root_mean_squared_error,
            X=X,
            y_true=y,
            response_method="predict",
            metric_name=f"RMSE {self._SCORE_OR_LOSS_ICONS['rmse']}",
            multioutput=multioutput,
            data_source_hash=data_source_hash,
            data_source=data_source,
        )

    def custom_metric(
        self,
        metric_function,
        response_method,
        *,
        metric_name=None,
        data_source="test",
        X=None,
        y=None,
        **kwargs,
    ):
        """Compute a custom metric.

        It brings some flexibility to compute any desired metric. However, we need to
        follow some rules:

        - `metric_function` should take `y_true` and `y_pred` as the first two
          positional arguments.
        - `response_method` corresponds to the estimator's method to be invoked to get
          the predictions. It can be a string or a list of strings to defined in which
          order the methods should be invoked.

        Parameters
        ----------
        metric_function : callable
            The metric function to be computed. The expected signature is
            `metric_function(y_true, y_pred, **kwargs)`.

        response_method : str or list of str
            The estimator's method to be invoked to get the predictions. The possible
            values are: `predict`, `predict_proba`, `predict_log_proba`, and
            `decision_function`.

        metric_name : str, default=None
            The name of the metric. If not provided, it will be inferred from the
            metric function.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the reporter.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the reporter.

        **kwargs : dict
            Any additional keyword arguments to be passed to the metric function.

        Returns
        -------
        pd.DataFrame
            The custom metric.
        """
        X, y, data_source_hash = self._get_X_y_and_data_source_hash(
            data_source=data_source, X=X, y=y
        )

        return self._compute_metric_scores(
            metric_function,
            X=X,
            y_true=y,
            response_method=response_method,
            metric_name=metric_name,
            data_source_hash=data_source_hash,
            data_source=data_source,
            **kwargs,
        )

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def _sort_methods_for_help(self, methods):
        """Override sort method for metrics-specific ordering.

        In short, we display the `report_metrics` first and then the `custom_metric`.
        """

        def _sort_key(method):
            name = method[0]
            if name == "custom_metric":
                priority = 1
            elif name == "report_metrics":
                priority = 2
            else:
                priority = 0
            return priority, name

        return sorted(methods, key=_sort_key)

    def _format_method_name(self, name):
        """Override format method for metrics-specific naming."""
        method_name = f"{name}(...)"
        method_name = method_name.ljust(22)
        if self._SCORE_OR_LOSS_ICONS[name] in ("(↗︎)", "(↘︎)"):
            if self._SCORE_OR_LOSS_ICONS[name] == "(↗︎)":
                method_name += f"[cyan]{self._SCORE_OR_LOSS_ICONS[name]}[/cyan]"
                return method_name.ljust(43)
            else:  # (↘︎)
                method_name += f"[orange1]{self._SCORE_OR_LOSS_ICONS[name]}[/orange1]"
                return method_name.ljust(49)
        else:
            return method_name.ljust(29)

    def _get_methods_for_help(self):
        """Override to exclude the plot accessor from methods list."""
        methods = super()._get_methods_for_help()
        return [(name, method) for name, method in methods if name != "plot"]

    def _create_help_tree(self):
        """Override to include plot methods in a separate branch."""
        tree = super()._create_help_tree()

        # Add plot methods in a separate branch
        plot_branch = tree.add("[bold cyan].plot 🎨[/bold cyan]")
        plot_methods = self.plot._get_methods_for_help()
        plot_methods = self.plot._sort_methods_for_help(plot_methods)

        for name, method in plot_methods:
            displayed_name = self.plot._format_method_name(name)
            description = self.plot._get_method_description(method)
            plot_branch.add(f".{displayed_name}".ljust(27) + f"- {description}")

        return tree

    def _get_help_panel_title(self):
        return f"[bold cyan]{self._icon} Available metrics methods[/bold cyan]"

    def _get_help_tree_title(self):
        return "[bold cyan]reporter.metrics[/bold cyan]"
