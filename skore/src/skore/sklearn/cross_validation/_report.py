import inspect

import pandas as pd
from joblib import Parallel, delayed
from rich.progress import track
from sklearn.base import is_classifier
from sklearn.model_selection import check_cv
from sklearn.utils._indexing import _safe_indexing
from sklearn.utils.metaestimators import available_if

from skore.sklearn._estimator import EstimatorReport
from skore.sklearn._help import _HelpAccessorMixin, _HelpReportMixin
from skore.sklearn.find_ml_task import _find_ml_task
from skore.utils._accessor import _check_supported_ml_task, register_accessor


def _generate_estimator_report(estimator, X, y, train_indices, test_indices):
    return EstimatorReport(
        estimator,
        fit=True,
        X_train=_safe_indexing(X, train_indices),
        y_train=_safe_indexing(y, train_indices),
        X_test=_safe_indexing(X, test_indices),
        y_test=_safe_indexing(y, test_indices),
    )


class CrossValidationReport(_HelpReportMixin):
    """Reporter for cross-validation results.

    Parameters
    ----------
    estimator : estimator object
        Estimator to make report from.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of supervised learning.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    """

    def __init__(
        self,
        estimator,
        X,
        y=None,
        cv=None,
        n_jobs=None,
    ):
        cv = check_cv(cv, y, classifier=is_classifier(estimator))
        parallel = Parallel(n_jobs=n_jobs, return_as="generator_unordered")
        # do not split the data to take advantage of the memory mapping
        generator = parallel(
            delayed(_generate_estimator_report)(
                estimator,
                X,
                y,
                train_indices,
                test_indices,
            )
            for train_indices, test_indices in cv.split(X, y)
        )

        n_splits = cv.get_n_splits(X, y)

        self.cv_results = list(
            track(generator, total=n_splits, description="Processing cross-validation")
        )
        self._ml_task = _find_ml_task(y, estimator=self.cv_results[0].estimator)

    ####################################################################################
    # Methods for the help
    ####################################################################################

    def _get_help_title(self):
        return (
            f"📓 Cross-validation Reporter\n"
            f"🔧 Available tools for diagnosing {self.cv_results[0].estimator_name} "
            "estimator"
        )


########################################################################################
# Base class for the accessors
########################################################################################


class _BaseAccessor(_HelpAccessorMixin):
    """Base class for all accessors."""

    def __init__(self, parent, icon):
        self._parent = parent
        self._icon = icon


########################################################################################
# Plotting accessor
########################################################################################


@register_accessor("plot", CrossValidationReport)
class _PlotAccessor(_BaseAccessor):
    def __init__(self, parent):
        super().__init__(parent, icon="🎨")


###############################################################################
# Metrics accessor
###############################################################################


@register_accessor("metrics", CrossValidationReport)
class _MetricsAccessor(_BaseAccessor):
    _SCORE_OR_LOSS_ICONS = {
        "accuracy": "📈",
        "precision": "📈",
        "recall": "📈",
        "brier_score": "📉",
        "roc_auc": "📈",
        "log_loss": "📉",
        "r2": "📈",
        "rmse": "📉",
        "report_metrics": "📈/📉",
        "custom_metric": "📈/📉",
    }

    def __init__(self, parent):
        super().__init__(parent, icon="📏")

    # TODO: should build on the `add_scorers` function
    def report_metrics(
        self,
        *,
        scoring=None,
        pos_label=1,
        scoring_kwargs=None,
    ):
        """Report a set of metrics for our estimator.

        Parameters
        ----------
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
                scoring = ["precision", "recall", "roc_auc", "brier_score"]
            elif self._parent._ml_task == "multiclass-classification":
                scoring = ["precision", "recall", "roc_auc"]
                if hasattr(self._parent._estimator, "predict_proba"):
                    scoring.append("log_loss")
            else:
                scoring = ["r2", "rmse"]

        scores = []

        for metric in scoring:
            # NOTE: we have to check specifically fort `_BaseScorer` first because this
            # is also a callable but it has a special private API that we can leverage
            # if isinstance(metric, _BaseScorer):
            #     # scorers have the advantage to have scoped defined kwargs
            #     metric_fn = partial(
            #         self.custom_metric,
            #         metric_function=metric._score_func,
            #         response_method=metric._response_method,
            #     )
            #     # forward the additional parameters specific to the scorer
            #     metrics_kwargs = {**metric._kwargs}
            if isinstance(metric, str) or callable(metric):
                if isinstance(metric, str):
                    metric_fn = getattr(self, metric)
                    metrics_kwargs = {}
                # else:
                #     metric_fn = partial(self.custom_metric, metric_function=metric)
                #     if scoring_kwargs is None:
                #         metrics_kwargs = {}
                #     else:
                #         # check if we should pass any parameters specific to the
                #         # metric callable
                #         metric_callable_params = inspect.signature(metric).parameters
                #         metrics_kwargs = {
                #             param: scoring_kwargs[param]
                #             for param in metric_callable_params
                #             if param in scoring_kwargs
                #         }
                metrics_params = inspect.signature(metric_fn).parameters
                if scoring_kwargs is not None:
                    for param in metrics_params:
                        if param in scoring_kwargs:
                            metrics_kwargs[param] = scoring_kwargs[param]
                if "pos_label" in metrics_params:
                    metrics_kwargs["pos_label"] = pos_label
            else:
                raise ValueError(
                    f"Invalid type of metric: {type(metric)} for metric: {metric}"
                )

            scores.append(metric_fn(**metrics_kwargs))

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
        df = pd.concat(
            [
                estimator_report.metrics.accuracy()
                for estimator_report in self._parent.cv_results
            ],
            keys=[f"Fold #{i}" for i in range(len(self._parent.cv_results))],
        )
        return df.swaplevel(0, 1)

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def precision(self, *, average="auto", pos_label=1):
        """Compute the precision score.

        Parameters
        ----------
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
        if average == "auto":
            if self._parent._ml_task == "binary-classification":
                average = "binary"
            else:
                average = "weighted"

        if average != "binary":
            # overwrite `pos_label` because it will be ignored
            pos_label = None

        df = pd.concat(
            [
                estimator_report.metrics.precision(average=average, pos_label=pos_label)
                for estimator_report in self._parent.cv_results
            ],
            keys=[f"Fold #{i}" for i in range(len(self._parent.cv_results))],
        )
        return df.swaplevel(0, 1)

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def recall(self, *, average="auto", pos_label=1):
        """Compute the recall score.

        Parameters
        ----------
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
        if average == "auto":
            if self._parent._ml_task == "binary-classification":
                average = "binary"
            else:
                average = "weighted"

        if average != "binary":
            # overwrite `pos_label` because it will be ignored
            pos_label = None

        df = pd.concat(
            [
                estimator_report.metrics.recall(average=average, pos_label=pos_label)
                for estimator_report in self._parent.cv_results
            ],
            keys=[f"Fold #{i}" for i in range(len(self._parent.cv_results))],
        )
        return df.swaplevel(0, 1)

    @available_if(
        _check_supported_ml_task(supported_ml_tasks=["binary-classification"])
    )
    def brier_score(self, *, pos_label=1):
        """Compute the Brier score.

        Parameters
        ----------
        pos_label : int, default=1
            The positive class.

        Returns
        -------
        pd.DataFrame
            The Brier score.
        """
        df = pd.concat(
            [
                estimator_report.metrics.brier_score(pos_label=pos_label)
                for estimator_report in self._parent.cv_results
            ],
            keys=[f"Fold #{i}" for i in range(len(self._parent.cv_results))],
        )
        return df.swaplevel(0, 1)

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def roc_auc(self, *, average="auto", multi_class="ovr"):
        """Compute the ROC AUC score.

        Parameters
        ----------
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
        if average == "auto":
            if self._parent._ml_task == "binary-classification":
                average = "macro"
            else:
                average = "weighted"

        df = pd.concat(
            [
                estimator_report.metrics.roc_auc(
                    average=average, multi_class=multi_class
                )
                for estimator_report in self._parent.cv_results
            ],
            keys=[f"Fold #{i}" for i in range(len(self._parent.cv_results))],
        )
        return df.swaplevel(0, 1)

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
        df = pd.concat(
            [
                estimator_report.metrics.log_loss()
                for estimator_report in self._parent.cv_results
            ],
            keys=[f"Fold #{i}" for i in range(len(self._parent.cv_results))],
        )
        return df.swaplevel(0, 1)

    @available_if(_check_supported_ml_task(supported_ml_tasks=["regression"]))
    def r2(self, *, multioutput="uniform_average"):
        """Compute the R² score.

        Parameters
        ----------
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
        df = pd.concat(
            [
                estimator_report.metrics.r2(multioutput=multioutput)
                for estimator_report in self._parent.cv_results
            ],
            keys=[f"Fold #{i}" for i in range(len(self._parent.cv_results))],
        )
        return df.swaplevel(0, 1)

    @available_if(_check_supported_ml_task(supported_ml_tasks=["regression"]))
    def rmse(self, *, multioutput="uniform_average"):
        """Compute the root mean squared error.

        Parameters
        ----------
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
        df = pd.concat(
            [
                estimator_report.metrics.rmse(multioutput=multioutput)
                for estimator_report in self._parent.cv_results
            ],
            keys=[f"Fold #{i}" for i in range(len(self._parent.cv_results))],
        )
        return df.swaplevel(0, 1)
