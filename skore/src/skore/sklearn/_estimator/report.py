import inspect
import time
from itertools import product

import joblib
import numpy as np
from rich.progress import track
from rich.tree import Tree
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils._response import _check_response_method, _get_response_values
from sklearn.utils.validation import check_is_fitted

from skore.externals._sklearn_compat import is_clusterer
from skore.sklearn._estimator.base import _BaseAccessor, _HelpMixin
from skore.sklearn.find_ml_task import _find_ml_task
from skore.utils._accessor import DirNamesMixin


class EstimatorReport(_HelpMixin, DirNamesMixin):
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

    Attributes
    ----------
    metrics : _MetricsAccessor
        Accessor for metrics-related operations.

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
