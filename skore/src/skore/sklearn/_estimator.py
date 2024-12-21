import inspect
import time
from io import StringIO

import numpy as np
import pandas as pd
from rich.console import Console
from rich.tree import Tree
from sklearn import metrics
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
            # sort the methods by the name of the method, except for the report_stats
            # method which should be at the start (special case for the metrics)
            if accessor_name == "metrics":
                # force to have the `report_stats` method at the start
                methods = sorted(methods, key=lambda x: x[0] != "report_stats")
            else:
                methods = sorted(methods)
            for name, method in methods:
                if not name.startswith("_") and not name.startswith("__"):
                    if accessor_name == "metrics":
                        # add the icon at the front of the metrics or loss
                        displayed_name = f"{IS_SCORE_OR_LOSS[name]} {name}"
                    else:
                        displayed_name = name
                    doc = method.__doc__.split("\n")[0]
                    branch.add(f"{displayed_name} - {doc}")

        _add_accessor_methods_to_tree(tree, self.plot, "🎨", "plot")
        _add_accessor_methods_to_tree(tree, self.metrics, "📏", "metrics")
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


###############################################################################
# Plotting accessor
###############################################################################


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


###############################################################################
# Metrics accessor
###############################################################################

IS_SCORE_OR_LOSS = {
    "accuracy": "📈",
    "precision": "📈",
    "recall": "📈",
    "brier_score": "📉",
    "roc_auc": "📈",
    "log_loss": "📉",
    "r2": "📈",
    "rmse": "📉",
    "report_stats": "📈/📉",
}


@register_accessor("metrics", EstimatorReport)
class _MetricsAccessor:
    def __init__(self, parent):
        self._parent = parent

    # TODO: should build on the `add_scorers` function
    def report_stats(self, scoring=None, positive_class=1):
        """Report a set of statistics for the metrics.

        Parameters
        ----------
        scoring : list of str, default=None
            The metrics to report.
        positive_class : int, default=1
            The positive class.

        Returns
        -------
        pd.DataFrame
            The statistics for the metrics.
        """
        if scoring is None:
            # Equivalent to _get_scorers_to_add
            if self._parent._ml_task == "binary-classification":
                scoring = ["precision", "recall", "roc_auc", "brier_score"]
            elif self._parent._ml_task == "multiclass-classification":
                scoring = ["precision", "recall", "roc_auc"]
                if hasattr(self._parent.cv_results["estimator"][0], "predict_proba"):
                    scoring.append("log_loss")
            else:
                scoring = ["r2", "rmse"]

        scores = []

        for metric in scoring:
            metric_fn = getattr(self, metric)

            if "positive_class" in inspect.signature(metric_fn).parameters:
                scores.append(metric_fn(positive_class=positive_class))
            else:
                scores.append(metric_fn())

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
                    scores[i].columns = pd.MultiIndex.from_tuples(
                        [(col, "") for col in score.columns]
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
        **metric_kwargs,
    ):
        y_pred = self._parent._get_cached_response_values(
            estimator_hash=self._parent._hash,
            estimator=self._parent.estimator,
            X=X,
            response_method=response_method,
            pos_label=pos_label,
        )

        cache_key = (self._parent._hash, metric_name)
        metric_params = inspect.signature(metric_fn).parameters
        if "pos_label" in metric_params:
            cache_key += (pos_label,)
        if "average" in metric_params:
            cache_key += (metric_kwargs["average"],)

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
                columns = [metric_name]
            else:
                classes = self._parent._estimator.classes_
                columns = [[metric_name] * len(classes), classes]
        elif self._parent._ml_task == "regression":
            if len(score) == 1:
                columns = [metric_name]
            else:
                columns = [
                    [metric_name] * len(score),
                    [f"Output #{i}" for i in range(len(score))],
                ]
        else:
            columns = None
        return pd.DataFrame(score, columns=columns, index=[self._parent.estimator_name])

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def accuracy(self):
        """Compute the accuracy score.

        Returns
        -------
        pd.DataFrame
            The accuracy score.
        """
        return self._compute_metric_scores(
            metrics.accuracy_score,
            X=self._parent._X_val,
            y_true=self._parent._y_val,
            response_method="predict",
            metric_name=f"Accuracy {IS_SCORE_OR_LOSS['accuracy']}",
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def precision(self, average="default", positive_class=1):
        """Compute the precision score.

        Parameters
        ----------
        average : {"default", "macro", "micro", "weighted", "samples"} or None, \
                default="default"
            The average to compute the precision score. By default, the average is
            "binary" for binary classification and "weighted" for multiclass
            classification.

        positive_class : int, default=1
            The positive class.

        Returns
        -------
        pd.DataFrame
            The precision score.
        """
        if average == "default":
            if self._parent._ml_task == "binary-classification":
                average = "binary"
            else:
                average = "weighted"

        return self._compute_metric_scores(
            metrics.precision_score,
            X=self._parent._X_val,
            y_true=self._parent._y_val,
            response_method="predict",
            pos_label=positive_class,
            metric_name=f"Precision {IS_SCORE_OR_LOSS['precision']}",
            average=average,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def recall(self, average="default", positive_class=1):
        """Compute the recall score.

        Parameters
        ----------
        average : {"default", "macro", "micro", "weighted", "samples"} or None, \
                default="default"
            The average to compute the recall score. By default, the average is
            "binary" for binary classification and "weighted" for multiclass
            classification.
        positive_class : int, default=1
            The positive class.

        Returns
        -------
        pd.DataFrame
            The recall score.
        """
        if average == "default":
            if self._parent._ml_task == "binary-classification":
                average = "binary"
            else:
                average = "weighted"

        return self._compute_metric_scores(
            metrics.recall_score,
            X=self._parent._X_val,
            y_true=self._parent._y_val,
            response_method="predict",
            pos_label=positive_class,
            metric_name=f"Recall {IS_SCORE_OR_LOSS['recall']}",
            average=average,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def brier_score(self, positive_class=1):
        """Compute the Brier score.

        Parameters
        ----------
        positive_class : int, default=1
            The positive class.

        Returns
        -------
        pd.DataFrame
            The Brier score.
        """
        return self._compute_metric_scores(
            metrics.brier_score_loss,
            X=self._parent._X_val,
            y_true=self._parent._y_val,
            response_method="predict_proba",
            metric_name=f"Brier score {IS_SCORE_OR_LOSS['brier_score']}",
            pos_label=positive_class,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def roc_auc(self, average="default"):
        """Compute the ROC AUC score.

        Parameters
        ----------
        average : {"default", "macro", "micro", "weighted", "samples"}, \
                default="default"
            The average to compute the ROC AUC score. By default, the average is
            "macro" for binary classification and multiclass classification with
            probability predictions and "weighted" for multiclass classification
            with 1-vs-rest predictions.

        Returns
        -------
        pd.DataFrame
            The ROC AUC score.
        """
        if average == "default":
            if self._parent._ml_task == "binary-classification":
                average = "macro"
                multi_class = "raise"
            else:
                average = "weighted"
                multi_class = "ovr"  # FIXME: do we expose it or not?

        return self._compute_metric_scores(
            metrics.roc_auc_score,
            X=self._parent._X_val,
            y_true=self._parent._y_val,
            response_method=["predict_proba", "decision_function"],
            metric_name=f"ROC AUC {IS_SCORE_OR_LOSS['roc_auc']}",
            average=average,
            multi_class=multi_class,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def log_loss(self):
        """Compute the log loss.

        Returns
        -------
        pd.DataFrame
            The log-loss.
        """
        return self._compute_metric_scores(
            metrics.log_loss,
            X=self._parent._X_val,
            y_true=self._parent._y_val,
            response_method="predict_proba",
            metric_name=f"Log loss {IS_SCORE_OR_LOSS['log_loss']}",
        )

    @available_if(_check_supported_ml_task(supported_ml_tasks=["regression"]))
    def r2(self):
        """Compute the R² score.

        Returns
        -------
        pd.DataFrame
            The R² score.
        """
        return self._compute_metric_scores(
            metrics.r2_score,
            X=self._parent._X_val,
            y_true=self._parent._y_val,
            response_method="predict",
            metric_name=f"R² {IS_SCORE_OR_LOSS['r2']}",
        )

    @available_if(_check_supported_ml_task(supported_ml_tasks=["regression"]))
    def rmse(self):
        """Compute the root mean squared error.

        Returns
        -------
        pd.DataFrame
            The root mean squared error.
        """
        return self._compute_metric_scores(
            metrics.root_mean_squared_error,
            X=self._parent._X_val,
            y_true=self._parent._y_val,
            response_method="predict",
            metric_name=f"RMSE {IS_SCORE_OR_LOSS['rmse']}",
        )
