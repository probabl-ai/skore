"""cross_validate function.

This function implements a wrapper over scikit-learn's
`cross_validate <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate>`_
function in order to enrich it with more information and enable more analysis.
"""

import contextlib
import inspect
import time
from typing import Literal, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.tree import Tree
from sklearn import metrics
from sklearn.linear_model._base import LinearModel
from sklearn.pipeline import Pipeline
from sklearn.utils._indexing import _safe_indexing
from sklearn.utils._response import _check_response_method, _get_response_values
from sklearn.utils.metaestimators import available_if

from skore.item.cross_validation_item import (
    CrossValidationAggregationItem,
    CrossValidationItem,
)
from skore.project import Project
from skore.sklearn._plot import RocCurveDisplay, WeightsDisplay


# TODO: this is really hacky, find a better solution
def _patch_pandas_repr_html():
    """Monkey patch pandas DataFrame _repr_html_ to apply custom styling."""
    original_repr_html = pd.DataFrame._repr_html_

    def new_repr_html(self):
        html = original_repr_html(self)
        if not isinstance(self.columns, pd.MultiIndex):
            styled_df = self.style.apply_index(
                _color_columns, axis="columns", level=[0]
            )
            return styled_df._repr_html_()
        return html

    pd.DataFrame._repr_html_ = new_repr_html


# Apply the monkey patch
_patch_pandas_repr_html()


def _find_ml_task(
    estimator, y
) -> Literal[
    "binary-classification",
    "multiclass-classification",
    "regression",
    "clustering",
    "unknown",
]:
    """Guess the ML task being addressed based on an estimator and a target array.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        An estimator.
    y : numpy.ndarray
        A target vector.

    Returns
    -------
    Literal["classification", "regression", "clustering", "unknown"]
        The guess of the kind of ML task being performed.
    """
    import sklearn.utils.multiclass
    from sklearn.base import is_classifier, is_regressor

    if y is None:
        # NOTE: The task might not be clustering
        return "clustering"

    if is_regressor(estimator):
        return "regression"

    type_of_target = sklearn.utils.multiclass.type_of_target(y)

    if is_classifier(estimator):
        if type_of_target == "binary":
            return "binary-classification"

        if type_of_target == "multiclass":
            return "multiclass-classification"

    if type_of_target == "unknown":
        return "unknown"

    if "continuous" in type_of_target:
        return "regression"

    return "classification"


def _get_scorers_to_add(estimator, y) -> list[str]:
    """Get a list of scorers based on `estimator` and `y`.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        An estimator.
    y : numpy.ndarray
        A target vector.

    Returns
    -------
    scorers_to_add : list[str]
        A list of scorers
    """
    ml_task = _find_ml_task(estimator, y)

    # Add scorers based on the ML task
    if ml_task == "regression":
        return ["r2", "neg_root_mean_squared_error"]
    if ml_task == "binary-classification":
        return ["roc_auc", "neg_brier_score", "recall", "precision"]
    if ml_task == "multiclass-classification":
        if hasattr(estimator, "predict_proba"):
            return [
                "recall_weighted",
                "precision_weighted",
                "roc_auc_ovr_weighted",
                "neg_log_loss",
            ]
        return ["recall_weighted", "precision_weighted"]
    return []


def _add_scorers(scorers, scorers_to_add):
    """Expand `scorers` with more scorers.

    The type of the resulting scorers object is dependent on the type of the input
    scorers:
    - If `scorers` is a dict, then extra scorers are added to the dict;
    - If `scorers` is a string or None, then it is converted to a dict and extra scorers
    are added to the dict;
    - If `scorers` is a list or tuple, then it is converted to a dict and extra scorers
    are added to the dict;
    - If `scorers` is a callable, then a new callable is created that
    returns a dict with the user-defined score as well as the scorers to add.
    In case the user-defined dict contains a metric with a name conflicting with the
    metrics we add, the user-defined metric always wins.

    Parameters
    ----------
    scorers : any type that is accepted by scikit-learn's cross_validate
        The scorer(s) to expand.
    scorers_to_add : list[str]
        The scorers to be added.

    Returns
    -------
    new_scorers : dict or callable
        The scorers after adding `scorers_to_add`.
    added_scorers : Iterable[str]
        The scorers that were actually added (i.e. the ones that were not already
        in `scorers`).
    """
    if scorers is None or isinstance(scorers, str):
        new_scorers, added_scorers = _add_scorers({"score": scorers}, scorers_to_add)
    elif isinstance(scorers, (list, tuple)):
        new_scorers, added_scorers = _add_scorers(
            {s: s for s in scorers}, scorers_to_add
        )
    elif isinstance(scorers, dict):
        new_scorers = {s: s for s in scorers_to_add} | scorers
        added_scorers = set(scorers_to_add) - set(scorers)
    elif callable(scorers):
        from sklearn.metrics import check_scoring
        from sklearn.metrics._scorer import _MultimetricScorer

        internal_scorer = _MultimetricScorer(
            scorers={
                s: check_scoring(estimator=None, scoring=s) for s in scorers_to_add
            }
        )

        def new_scorer(estimator, X, y) -> dict:
            scores = scorers(estimator, X, y)
            if isinstance(scores, dict):
                return internal_scorer(estimator, X, y) | scores
            return internal_scorer(estimator, X, y) | {"score": scores}

        new_scorers = new_scorer

        # In this specific case, we can't know if there is overlap between the
        # user-defined scores and ours, so we take the least risky option
        # which is to say we added nothing; that way, we won't remove anything
        # after cross-validation is computed
        added_scorers = []

    return new_scorers, added_scorers


def _strip_cv_results_scores(cv_results: dict, added_scorers: list[str]) -> dict:
    """Remove information about `added_scorers` in `cv_results`.

    Parameters
    ----------
    cv_results : dict
        A dict of the form returned by scikit-learn's cross_validate function.
    added_scorers : list[str]
        A list of scorers in `cv_results` which should be removed.

    Returns
    -------
    dict
        A new cv_results dict, with the specified scorers information removed.
    """
    # Takes care both of train and test scores
    return {
        k: v
        for k, v in cv_results.items()
        if not any(added_scorer in k for added_scorer in added_scorers)
    }


def cross_validate(*args, project: Optional[Project] = None, **kwargs) -> dict:
    """Evaluate estimator by cross-validation and output UI-friendly object.

    This function wraps scikit-learn's :func:`~sklearn.model_selection.cross_validate`
    function, to provide more context and facilitate the analysis.
    As such, the arguments are the same as scikit-learn's ``cross_validate`` function.

    The dict returned by this function is a strict super-set of the one returned by
    scikit-learn's :func:`~sklearn.model_selection.cross_validate`.

    For a user guide and in-depth example, see :ref:`example_cross_validate`.

    Parameters
    ----------
    *args
        Positional arguments accepted by scikit-learn's
        :func:`~sklearn.model_selection.cross_validate`,
        such as ``estimator`` and ``X``.
    project : Project, optional
        A project to save cross-validation data into. If None, no save is performed.
    **kwargs
        Additional keyword arguments accepted by scikit-learn's
        :func:`~sklearn.model_selection.cross_validate`.

    Returns
    -------
    cv_results : dict
        A dict of the form returned by scikit-learn's
        :func:`~sklearn.model_selection.cross_validate` function.

    Examples
    --------
    >>> def prepare_cv():
    ...     from sklearn import datasets, linear_model
    ...     diabetes = datasets.load_diabetes()
    ...     X = diabetes.data[:150]
    ...     y = diabetes.target[:150]
    ...     lasso = linear_model.Lasso()
    ...     return lasso, X, y

    >>> project = skore.load("project.skore")  # doctest: +SKIP
    >>> lasso, X, y = prepare_cv()  # doctest: +SKIP
    >>> cross_validate(lasso, X, y, cv=3, project=project)  # doctest: +SKIP
    {'fit_time': array(...), 'score_time': array(...), 'test_score': array(...)}
    """
    import sklearn.model_selection

    # Recover specific arguments
    estimator = args[0] if len(args) >= 1 else kwargs.get("estimator")
    X = args[1] if len(args) >= 2 else kwargs.get("X")
    y = args[2] if len(args) == 3 else kwargs.get("y")

    try:
        scorers = kwargs.pop("scoring")
    except KeyError:
        scorers = None

    # Extend scorers with other relevant scorers
    scorers_to_add = _get_scorers_to_add(estimator, y)
    new_scorers, added_scorers = _add_scorers(scorers, scorers_to_add)

    cv_results = sklearn.model_selection.cross_validate(
        *args, **kwargs, scoring=new_scorers
    )

    cross_validation_item = CrossValidationItem.factory(cv_results, estimator, X, y)

    if project is not None:
        try:
            cv_results_history = project.get_item_versions("cross_validation")
        except KeyError:
            cv_results_history = []

        agg_cross_validation_item = CrossValidationAggregationItem.factory(
            cv_results_history + [cross_validation_item]
        )

        project.put_item("cross_validation_aggregated", agg_cross_validation_item)
        project.put_item("cross_validation", cross_validation_item)

    # If in a IPython context (e.g. Jupyter notebook), display the plot
    with contextlib.suppress(ImportError):
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.display import display

        if InteractiveShell.initialized():
            display(cross_validation_item.plot)

    # Remove information related to our scorers, so that our return value is
    # the same as sklearn's
    stripped_cv_results = _strip_cv_results_scores(cv_results, added_scorers)

    # Add explicit metric to result (rather than just "test_score")
    if isinstance(scorers, str):
        if kwargs.get("return_train_score") is not None:
            stripped_cv_results[f"train_{scorers}"] = stripped_cv_results["train_score"]
        stripped_cv_results[f"test_{scorers}"] = stripped_cv_results["test_score"]

    return stripped_cv_results


def register_accessor(name, target_cls):
    """Register an accessor for a class.

    Parameters
    ----------
    name : str
        The name of the accessor.
    target_cls : type
        The class to register the accessor for.
    """

    def decorator(accessor_cls):
        def getter(self):
            attr = f"_accessor_{accessor_cls.__name__}"
            if not hasattr(self, attr):
                setattr(self, attr, accessor_cls(self))
            return getattr(self, attr)

        setattr(target_cls, name, property(getter))
        return accessor_cls

    return decorator


class CrossValidationReporter:
    """Analyse the output of scikit-learn's cross_validate function.

    Parameters
    ----------
    cv_results : dict
        A dict of the form returned by scikit-learn's
        :func:`~sklearn.model_selection.cross_validate` function.
    data : {array-like, sparse matrix}
        The data used to fit the estimator.
    target : array-like
        The target vector.
    sample_weight : array-like, default=None
        The sample weights.
    """

    def __init__(self, cv_results, data, target, sample_weight=None):
        required_keys = {"estimator": "return_estimator", "indices": "return_indices"}
        missing_keys = [key for key in required_keys if key not in cv_results]

        if missing_keys:
            missing_params = [f"{required_keys[key]}=True" for key in missing_keys]
            raise RuntimeError(
                f"The keys {missing_keys} are required in `cv_results` to create a "
                f"`MetricsReporter` instance, but they are not found. You need "
                f"to set {', '.join(missing_params)} in "
                "`sklearn.model_selection.cross_validate()`."
            )

        self.cv_results = cv_results
        self.data = data
        self.target = target
        self.sample_weight = sample_weight

        self._rng = np.random.default_rng(time.time_ns())
        # It could be included in cv_results but let's avoid to mutate it for the moment
        self._hash = [
            self._rng.integers(low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max)
            for _ in range(len(cv_results["estimator"]))
        ]
        self._cache = {}
        self._ml_task = _find_ml_task(cv_results["estimator"][0], target)

    def _get_cached_response_values(
        self,
        *,
        hash,
        estimator,
        X,
        response_method,
        pos_label=None,
    ):
        prediction_method = _check_response_method(estimator, response_method).__name__
        if prediction_method in ("predict_proba", "decision_function"):
            # pos_label is only important in classification and with probabilities
            # and decision functions
            cache_key = (hash, pos_label, prediction_method)
        else:
            cache_key = (hash, prediction_method)

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

    def help(self):
        """Display available plotting and metrics functions using rich."""
        console = Console()
        tree = Tree("🔧 Available tools with this cross-validation reporter")

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
                    branch.add(f"[green]{name}[/green] - {doc}")

        # Add methods for each accessor
        _add_accessor_methods_to_tree(tree, self.plot, "🎨", "plot")
        _add_accessor_methods_to_tree(tree, self.metrics, "📏", "metrics")
        _add_accessor_methods_to_tree(tree, self.inspection, "🔍", "inspection")

        console.print(tree)


def _check_supported_ml_task(supported_ml_tasks):
    def check(accessor):
        supported_task = any(
            task in accessor._parent._ml_task for task in supported_ml_tasks
        )

        if not supported_task:
            raise AttributeError(
                f"The {accessor._parent._ml_task} task is not a supported task by "
                f"function called. The supported tasks are {supported_ml_tasks}."
            )

        return True

    return check


def _check_supported_estimator(supported_estimators):
    def check(accessor):
        estimators = accessor._parent.cv_results["estimator"]
        if isinstance(estimators[0], Pipeline):
            estimators = [est.steps[-1][1] for est in estimators]
        supported_estimator = isinstance(estimators[0], supported_estimators)

        if not supported_estimator:
            raise AttributeError(
                f"The {estimators[0].__class__.__name__} estimator is not supported "
                "by the function called."
            )

        return True

    return check


@register_accessor("plot", CrossValidationReporter)
class _PlotAccessor:
    def __init__(self, parent):
        self._parent = parent

    @available_if(
        _check_supported_ml_task(supported_ml_tasks=["binary-classification"])
    )
    def roc(
        self,
        positive_class=None,
        ax=None,
        name=None,
        plot_chance_level=True,
        chance_level_kw=None,
        despine=True,
        backend="matplotlib",
    ):
        """Plot the ROC curve.

        Parameters
        ----------
        positive_class : str, default=None
            The positive class.
        ax : matplotlib.axes.Axes, default=None
            The axes to plot on.
        name : str, default=None
            The name of the plot.
        plot_chance_level : bool, default=True
            Whether to plot the chance level.
        chance_level_kw : dict, default=None
            The keyword arguments for the chance level.
        despine : bool, default=True
            Whether to despine the plot. Only relevant for matplotlib backend.
        backend : {"matplotlib", "plotly"}, default="matplotlib"
            The backend to use for plotting.

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
            The ROC curve plot.
        """
        prediction_method = ["predict_proba", "decision_function"]

        for fold_idx, (hash, estimator, test_indices) in enumerate(
            zip(
                self._parent._hash,
                self._parent.cv_results["estimator"],
                self._parent.cv_results["indices"]["test"],
            )
        ):
            y_pred = self._parent._get_cached_response_values(
                hash=hash,
                estimator=estimator,
                X=self._parent.data,
                response_method=prediction_method,
                pos_label=positive_class,
            )

            y_true_split = _safe_indexing(self._parent.target, test_indices)
            y_pred_split = _safe_indexing(y_pred, test_indices)
            if self._parent.sample_weight is not None:
                sample_weight_split = _safe_indexing(
                    self._parent.sample_weight, test_indices
                )
            else:
                sample_weight_split = None

            cache_key = (hash, RocCurveDisplay.__name__)

            # trick to have the chance level only in the last plot
            if fold_idx == len(self._parent._hash) - 1:
                plot_chance_level_ = plot_chance_level
            else:
                plot_chance_level_ = False

            if name is None:
                name_ = f"{estimator.__class__.__name__} - Fold {fold_idx}"
            else:
                name_ = f"{name} - Fold {fold_idx}"

            if cache_key in self._parent._cache:
                display = self._parent._cache[cache_key].plot(
                    ax=ax,
                    backend=backend,
                    name=name_,
                    plot_chance_level=plot_chance_level_,
                    chance_level_kw=chance_level_kw,
                    despine=despine,
                )
            else:
                display = RocCurveDisplay.from_predictions(
                    y_true_split,
                    y_pred_split,
                    sample_weight=sample_weight_split,
                    pos_label=positive_class,
                    ax=ax,
                    backend=backend,
                    name=name_,
                    plot_chance_level=plot_chance_level_,
                    chance_level_kw=chance_level_kw,
                    despine=despine,
                )
                self._parent._cache[cache_key] = display

            # overwrite for the subsequent plots
            ax = display.ax_

        return display.figure_

    @available_if(_check_supported_estimator(supported_estimators=LinearModel))
    def model_weights(
        self, *, ax=None, style="boxplot", add_data_points=True, backend="matplotlib"
    ):
        """Plot the model weights.

        This is only available for linear models with a `coef_` attribute.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            The axes to plot on.
        style : {"boxplot", "violinplot"}, default="boxplot"
            The style of the plot.
        add_data_points : bool, default=False
            Whether to add data points to the plot.
        backend : {"matplotlib", "plotly"}, default="matplotlib"
            The backend to use for plotting.

        Returns
        -------
        matplotlib.figure.Figure
            The model weights plot.
        """
        # no caching here since there is not computation to do
        display = WeightsDisplay.from_cv_results(
            self._parent.cv_results,
            ax=ax,
            style=style,
            add_data_points=add_data_points,
            backend=backend,
        )
        return display.figure_


# losses are red, scores are green
METRIC_COLORS = {
    "Accuracy": "green",
    "Precision": "green",
    "Recall": "green",
    "Brier score": "red",
    "ROC AUC": "green",
    "Log loss": "red",
    "R²": "green",
    "RMSE": "red",
}


def _color_columns(style):
    return [
        (f"background-color: {METRIC_COLORS[col]};" if col in METRIC_COLORS else "")
        for col in style
    ]


@register_accessor("metrics", CrossValidationReporter)
class _MetricsAccessor:
    def __init__(self, parent):
        self._parent = parent

    # TODO: should build on the `add_scorers` function
    def report_stats(self, scoring=None, positive_class=1):
        """Report statistics for the metrics.

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
        *,
        response_method,
        pos_label=None,
        metric_name=None,
        **metric_kwargs,
    ):
        scores = []

        for hash, estimator, test_indices in zip(
            self._parent._hash,
            self._parent.cv_results["estimator"],
            self._parent.cv_results["indices"]["test"],
        ):
            y_pred = self._parent._get_cached_response_values(
                hash=hash,
                estimator=estimator,
                X=self._parent.data,
                response_method=response_method,
                pos_label=pos_label,
            )

            cache_key = (hash, metric_name)
            metric_params = inspect.signature(metric_fn).parameters
            if "pos_label" in metric_params:
                cache_key += (pos_label,)
            if "average" in metric_params:
                cache_key += (metric_kwargs["average"],)

            if cache_key in self._parent._cache:
                score = self._parent._cache[cache_key]
            else:
                y_true_split = _safe_indexing(self._parent.target, test_indices)
                y_pred_split = _safe_indexing(y_pred, test_indices)

                sample_weight_split = None
                if self._parent.sample_weight is not None:
                    sample_weight_split = _safe_indexing(
                        self._parent.sample_weight, test_indices
                    )

                metric_params = inspect.signature(metric_fn).parameters
                kwargs = {**metric_kwargs}
                if "pos_label" in metric_params:
                    kwargs.update(pos_label=pos_label)

                score = metric_fn(
                    y_true_split,
                    y_pred_split,
                    sample_weight=sample_weight_split,
                    **kwargs,
                )

                self._parent._cache[cache_key] = score

            scores.append(score)
        scores = np.array(scores)

        metric_name = metric_name or metric_fn.__name__

        if self._parent._ml_task in [
            "binary-classification",
            "multiclass-classification",
        ]:
            if scores.ndim == 1:
                columns = [metric_name]
            else:
                classes = self._parent.cv_results["estimator"][0].classes_
                columns = [[metric_name] * len(classes), classes]
        elif self._parent._ml_task == "regression":
            if scores.ndim == 1:
                columns = [metric_name]
            else:
                columns = [
                    [metric_name] * scores.shape[1],
                    [f"Output #{i}" for i in range(scores.shape[1])],
                ]
        else:
            columns = None
        return pd.DataFrame(scores, columns=columns)

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
            metrics.accuracy_score, response_method="predict", metric_name="Accuracy"
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
            response_method="predict",
            pos_label=positive_class,
            metric_name="Precision",
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
            response_method="predict",
            pos_label=positive_class,
            metric_name="Recall",
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
            response_method="predict_proba",
            metric_name="Brier score",
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
            response_method=["predict_proba", "decision_function"],
            metric_name="ROC AUC",
            average=average,
            multi_class=multi_class,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def log_loss(self):
        """Compute the log loss score.

        Returns
        -------
        pd.DataFrame
            The log-loss.
        """
        return self._compute_metric_scores(
            metrics.log_loss, response_method="predict_proba", metric_name="Log loss"
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
            metrics.r2_score, response_method="predict", metric_name="R²"
        )

    @available_if(_check_supported_ml_task(supported_ml_tasks=["regression"]))
    def rmse(self):
        """Compute the RMSE score.

        Returns
        -------
        pd.DataFrame
            The RMSE score.
        """
        return self._compute_metric_scores(
            metrics.root_mean_squared_error,
            response_method="predict",
            metric_name="RMSE",
        )


@register_accessor("inspection", CrossValidationReporter)
class _InspectionAccessor:
    def __init__(self, parent):
        self._parent = parent

    # FIXME: we should some editorial choice here
    def feature_importances(self, type="default"):
        """Compute feature importances.

        Parameters
        ----------
        type : {"default", "mdi", "permutation", "weights"}
            The type of feature importances to compute. By default, we provide the
            weights for linear models and the mean decrease in impurity for tree-based
            models.

        Returns
        -------
        pd.DataFrame
            The feature importances.
        """
        estimators = self._parent.cv_results["estimator"]
        if isinstance(estimators[0], Pipeline):
            estimators = [est.steps[-1][1] for est in estimators]

        if type == "default":
            if hasattr(estimators[0], "coef_"):
                type = "weights"
            elif hasattr(estimators[0], "feature_importances_"):
                type = "mdi"
            else:
                raise ValueError(
                    "No default feature importances for "
                    f"{estimators[0].__class__.__name__}."
                )
        elif type == "permutation":
            # TODO: we need to compute the permutation importances.
            # We need good caching and good parameter here.
            # We should probably make it that this computation is shared with the
            # plotting since this is quite expensive in terms of computation.
            raise NotImplementedError(
                "Permutation importances are not yet implemented."
            )

        if hasattr(estimators[0], "feature_names_in_"):
            feature_names = estimators[0].feature_names_in_
        else:
            feature_names = [
                f"Feature #{i}" for i in range(estimators[0].n_features_in_)
            ]

        if type == "weights":
            importances = [est.coef_ for est in estimators]
        elif type == "mdi":
            importances = [est.feature_importances_ for est in estimators]
        else:
            raise NotImplementedError

        return pd.DataFrame(importances, columns=feature_names)
