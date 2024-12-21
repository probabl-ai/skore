import inspect
import time
from io import StringIO

import numpy as np
from rich.console import Console
from rich.tree import Tree
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils._response import _check_response_method, _get_response_values
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from skore.sklearn._plot import RocCurveDisplay
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


class EstimatorReport:
    """Report for a fitted estimator.

    This class provides a set of tools to quickly validate and inspect a fitted
    estimator.

    To quickly check the available tools, use the `help` method::

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(random_state=42)
    >>> X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)
    >>> estimator = LogisticRegression().fit(X_train, y_train)
    >>> from skore import EstimatorReport
    >>> report = EstimatorReport.from_fitted_estimator(estimator, X=X_val, y=y_val)
    >>> report.help()
    🔧 Available tools for LogisticRegression estimator
    ...

    Parameters
    ----------
    estimator : estimator object
        Fitted estimator to make report from.
    X_train : {array-like, sparse matrix} of shape (n_samples, n_features) or \
            None
        Training data.
    y_train : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        Training target.
    X_val : {array-like, sparse matrix} of shape (n_samples, n_features) or None
        Validation data. It should have the same structure as the training data.
    y_val : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        Validation target.
    """

    def __init__(self, estimator, *, X_train, y_train, X_val, y_val):
        check_is_fitted(estimator)

        # private storage to be able to invalidate the cache when the user alters
        # those attributes
        self._estimator = estimator
        self._X_train = X_train
        self._y_train = y_train
        self._X_val = X_val
        self._y_val = y_val

        self._initialize_state()

    def _initialize_state(self):
        """Initialize/reset the random number generator, hash, and cache."""
        self._rng = np.random.default_rng(time.time_ns())
        self._hash = self._rng.integers(
            low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max
        )
        self._cache = {}
        self._ml_task = _find_ml_task(self._y_val, estimator=self._estimator)

    # NOTE:
    # For the moment, we do not allow to alter the estimator and the training data.
    # For the validation set, we allow it and we invalidate the cache.

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        raise AttributeError(
            "The estimator attribute is immutable. "
            "Please use the `from_fitted_estimator` or `from_unfitted_estimator` "
            "methods to create a new report."
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
    def X_val(self):
        return self._X_val

    @X_val.setter
    def X_val(self, value):
        self._X_val = value
        self._initialize_state()

    @property
    def y_val(self):
        return self._y_val

    @y_val.setter
    def y_val(self, value):
        self._y_val = value
        self._initialize_state()

    @property
    def estimator_name(self):
        if isinstance(self._estimator, Pipeline):
            name = self._estimator[-1].__class__.__name__
        else:
            name = self._estimator.__class__.__name__
        return name

    @classmethod
    def from_fitted_estimator(cls, estimator, *, X, y=None):
        """Create an estimator report from a fitted estimator.

        Parameters
        ----------
        estimator : estimator object
            Fitted estimator to make report from.
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Validation data. It should have the same structure as the training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Validation target.

        Returns
        -------
        EstimatorReport
            The estimator report.
        """
        return cls(estimator=estimator, X_train=None, y_train=None, X_val=X, y_val=y)

    @classmethod
    def from_unfitted_estimator(
        cls, estimator, *, X_train, y_train=None, X_test=None, y_test=None
    ):
        """Create an estimator report from a fitted estimator.

        Parameters
        ----------
        estimator : estimator object
            The estimator that will be fitted on the training data. The estimator
            is cloned before being fitted.
        X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Training target.
        X_test : {array-like, sparse matrix} of shape (n_samples, n_features), \
                default=None
            Validation data. It should have the same structure as the training data.
        y_test : array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Validation target.

        Returns
        -------
        EstimatorReport
            The estimator report.
        """
        estimator = clone(estimator).fit(X_train, y_train)
        return cls(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
        )

    def _get_cached_response_values(
        self,
        *,
        estimator_hash,
        estimator,
        X,
        response_method,
        pos_label=None,
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

        Returns
        -------
        array-like of shape (n_samples,) or (n_samples, n_outputs)
            The response values.
        """
        prediction_method = _check_response_method(estimator, response_method).__name__
        if prediction_method in ("predict_proba", "decision_function"):
            # pos_label is only important in classification and with probabilities
            # and decision functions
            cache_key = (estimator_hash, pos_label, prediction_method)
        else:
            cache_key = (estimator_hash, prediction_method)

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

    def _create_help_tree(self):
        """Create a rich Tree with the available tools."""
        tree = Tree(
            f"🔧 Available tools for diagnosing {self.estimator_name} estimator"
        )

        def _add_accessor_methods_to_tree(tree, accessor, icon, accessor_name):
            branch = tree.add(f"{icon} {accessor_name}")
            methods = inspect.getmembers(accessor, predicate=inspect.ismethod)
            for name, method in methods:
                if not name.startswith("_") and not name.startswith("__"):
                    doc = (
                        method.__doc__.split("\n")[0]
                        if method.__doc__
                        else "No description available"
                    )
                    branch.add(f"{name} - {doc}")

        _add_accessor_methods_to_tree(tree, self.plot, "🎨", "plot")
        # _add_accessor_methods_to_tree(tree, self.metrics, "📏", "metrics")
        # _add_accessor_methods_to_tree(tree, self.inspection, "🔍", "inspection")
        # _add_accessor_methods_to_tree(tree, self.linting, "✔️", "linting")

        return tree

    def help(self):
        """Display available plotting and metrics functions using rich."""
        from skore import console  # avoid circular import

        console.print(self._create_help_tree())

    def __repr__(self):
        """Return a string representation of the EstimatorReport using rich."""
        # Create a string buffer to capture the tree output
        console = Console(file=StringIO(), force_terminal=False)
        console.print("📓 Estimator Reporter", self._create_help_tree())
        return console.file.getvalue()


@register_accessor("plot", EstimatorReport)
class _PlotAccessor:
    def __init__(self, parent):
        self._parent = parent

    @available_if(
        _check_supported_ml_task(supported_ml_tasks=["binary-classification"])
    )
    def roc(self, *, positive_class=None, ax=None, name=None):
        """Plot the ROC curve.

        Parameters
        ----------
        positive_class : str, default=None
            The positive class.
        ax : matplotlib.axes.Axes, default=None
            The axes to plot on.
        name : str, default=None
            The name of the plot.

        Returns
        -------
        RocCurveDisplay
            The ROC curve display.
        """
        prediction_method = ["predict_proba", "decision_function"]

        # FIXME: only computing on the validation set for now
        y_pred = self._parent._get_cached_response_values(
            estimator_hash=self._parent._hash,
            estimator=self._parent.estimator,
            X=self._parent.X_val,
            response_method=prediction_method,
            pos_label=positive_class,
        )

        cache_key = (self._parent._hash, RocCurveDisplay.__name__, positive_class)
        name_ = self._parent.estimator_name if name is None else name

        if cache_key in self._parent._cache:
            display = self._parent._cache[cache_key].plot(
                ax=ax,
                name=name_,
                plot_chance_level=True,
                despine=True,
            )
        else:
            display = RocCurveDisplay.from_predictions(
                self._parent.y_val,
                y_pred,
                pos_label=positive_class,
                ax=ax,
                name=name_,
                plot_chance_level=True,
                despine=True,
            )
            self._parent._cache[cache_key] = display

        return display
