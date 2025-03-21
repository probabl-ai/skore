from typing import Any, Callable, Literal, Optional, Union

import joblib
import numpy as np
import pandas as pd
from rich.progress import Progress
from sklearn.metrics import make_scorer
from sklearn.metrics._scorer import _BaseScorer as SKLearnScorer
from sklearn.utils.metaestimators import available_if

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseAccessor, _get_cached_response_values
from skore.sklearn._cross_validation.report import CrossValidationReport
from skore.sklearn._plot import (
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from skore.utils._accessor import _check_supported_ml_task
from skore.utils._index import flatten_multi_index
from skore.utils._parallel import Parallel, delayed
from skore.utils._progress_bar import progress_decorator

DataSource = Literal["test", "train"]


class _MetricsAccessor(_BaseAccessor["CrossValidationReport"], DirNamesMixin):
    """Accessor for metrics-related operations.

    You can access this accessor using the `metrics` attribute.
    """

    _SCORE_OR_LOSS_INFO: dict[str, dict[str, str]] = {
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

    def __init__(self, parent: CrossValidationReport) -> None:
        super().__init__(parent)

        self._progress_info: Optional[dict[str, Any]] = None
        self._parent_progress: Optional[Progress] = None

    def report_metrics(
        self,
        *,
        data_source: DataSource = "test",
        scoring: Optional[Union[list[str], Callable, SKLearnScorer]] = None,
        scoring_names: Optional[list[str]] = None,
        scoring_kwargs: Optional[dict[str, Any]] = None,
        pos_label: Optional[Union[int, float, bool, str]] = None,
        indicator_favorability: bool = False,
        flat_index: bool = False,
        aggregate: Optional[
            Union[Literal["mean", "std"], list[Literal["mean", "std"]]]
        ] = None,
    ) -> pd.DataFrame:
        """Report a set of metrics for our estimator.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        scoring : list of str, callable, or scorer, default=None
            The metrics to report. You can get the possible list of string by calling
            `report.metrics.help()`. When passing a callable, it should take as
            arguments `y_true`, `y_pred` as the two first arguments. Additional
            arguments can be passed as keyword arguments and will be forwarded with
            `scoring_kwargs`. If the callable API is too restrictive (e.g. need to pass
            same parameter name with different values), you can use scikit-learn scorers
            as provided by :func:`sklearn.metrics.make_scorer`.

        scoring_names : list of str, default=None
            Used to overwrite the default scoring names in the report. It should be of
            the same length as the `scoring` parameter.

        scoring_kwargs : dict, default=None
            The keyword arguments to pass to the scoring functions.

        pos_label : int, float, bool or str, default=None
            The positive class.

        indicator_favorability : bool, default=False
            Whether or not to add an indicator of the favorability of the metric as
            an extra column in the returned DataFrame.

        flat_index : bool, default=False
            Whether to flatten the `MultiIndex` columns. Flat index will always be lower
            case, do not include spaces and remove the hash symbol to ease indexing.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The statistics for the metrics.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        >>> report.metrics.report_metrics(
        ...     scoring=["precision", "recall"],
        ...     pos_label=1,
        ...     aggregate=["mean", "std"],
        ...     indicator_favorability=True,
        ... )
                  LogisticRegression           Favorability
                                mean       std
        Metric
        Precision           0.94...  0.02...         (↗︎)
        Recall              0.96...  0.02...         (↗︎)
        """
        results = self._compute_metric_scores(
            report_metric_name="report_metrics",
            data_source=data_source,
            aggregate=aggregate,
            scoring=scoring,
            pos_label=pos_label,
            scoring_kwargs=scoring_kwargs,
            scoring_names=scoring_names,
            indicator_favorability=indicator_favorability,
        )
        if flat_index:
            if isinstance(results.columns, pd.MultiIndex):
                results.columns = flatten_multi_index(results.columns)
            if isinstance(results.index, pd.MultiIndex):
                results.index = flatten_multi_index(results.index)
        return results

    @progress_decorator(description="Compute metric for each split")
    def _compute_metric_scores(
        self,
        report_metric_name: str,
        *,
        data_source: DataSource = "test",
        aggregate: Optional[
            Union[Literal["mean", "std"], list[Literal["mean", "std"]]]
        ] = None,
        **metric_kwargs: Any,
    ) -> pd.DataFrame:
        # build the cache key components to finally create a tuple that will be used
        # to check if the metric has already been computed
        cache_key_parts: list[Any] = [
            self._parent._hash,
            report_metric_name,
            data_source,
        ]
        if aggregate is None:
            cache_key_parts.append(aggregate)
        else:
            cache_key_parts.extend(tuple(aggregate))
        ordered_metric_kwargs = sorted(metric_kwargs.keys())
        for key in ordered_metric_kwargs:
            if isinstance(metric_kwargs[key], (np.ndarray, list, dict)):
                cache_key_parts.append(joblib.hash(metric_kwargs[key]))
            else:
                cache_key_parts.append(metric_kwargs[key])
        cache_key = tuple(cache_key_parts)

        assert self._progress_info is not None, "Progress info not set"
        progress = self._progress_info["current_progress"]
        main_task = self._progress_info["current_task"]

        total_estimators = len(self._parent.estimator_reports_)
        progress.update(main_task, total=total_estimators)

        if cache_key in self._parent._cache:
            results = self._parent._cache[cache_key]
        else:
            parallel = Parallel(
                n_jobs=self._parent.n_jobs,
                return_as="generator",
                require="sharedmem",
            )
            generator = parallel(
                delayed(getattr(report.metrics, report_metric_name))(
                    data_source=data_source, **metric_kwargs
                )
                for report in self._parent.estimator_reports_
            )
            results = []
            for result in generator:
                results.append(result)
                progress.update(main_task, advance=1, refresh=True)

            results = pd.concat(
                results,
                axis=1,
                keys=[f"Split #{i}" for i in range(len(results))],
            )
            results = results.swaplevel(0, 1, axis=1)

            # Pop the favorability column if it exists, to:
            # - not use it in the aggregate operation
            # - later to only report a single column and not by split columns
            if metric_kwargs.get("indicator_favorability", False):
                favorability = results.pop("Favorability").iloc[:, 0]
            else:
                favorability = None

            if aggregate:
                if isinstance(aggregate, str):
                    aggregate = [aggregate]

                results = results.aggregate(func=aggregate, axis=1)
                results = pd.concat(
                    [results], keys=[self._parent.estimator_name_], axis=1
                )

            if favorability is not None:
                results["Favorability"] = favorability

            self._parent._cache[cache_key] = results
        return results

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def accuracy(
        self,
        *,
        data_source: DataSource = "test",
        aggregate: Optional[
            Union[Literal["mean", "std"], list[Literal["mean", "std"]]]
        ] = None,
    ) -> pd.DataFrame:
        """Compute the accuracy score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The accuracy score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        >>> report.metrics.accuracy()
                    LogisticRegression
                                Split #0  Split #1
        Metric
        Accuracy                0.94...   0.94...
        """
        return self.report_metrics(
            scoring=["accuracy"],
            data_source=data_source,
            aggregate=aggregate,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def precision(
        self,
        *,
        data_source: DataSource = "test",
        average: Optional[
            Literal["binary", "macro", "micro", "weighted", "samples"]
        ] = None,
        pos_label: Optional[Union[int, float, bool, str]] = None,
        aggregate: Optional[
            Union[Literal["mean", "std"], list[Literal["mean", "std"]]]
        ] = None,
    ) -> pd.DataFrame:
        """Compute the precision score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        average : {"binary","macro", "micro", "weighted", "samples"} or None, \
                default=None
            Used with multiclass problems.
            If `None`, the metrics for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            - "binary": Only report results for the class specified by `pos_label`.
              This is applicable only if targets (`y_{true,pred}`) are binary.
            - "micro": Calculate metrics globally by counting the total true positives,
              false negatives and false positives.
            - "macro": Calculate metrics for each label, and find their unweighted
              mean.  This does not take label imbalance into account.
            - "weighted": Calculate metrics for each label, and find their average
              weighted by support (the number of true instances for each label). This
              alters 'macro' to account for label imbalance; it can result in an F-score
              that is not between precision and recall.
            - "samples": Calculate metrics for each instance, and find their average
              (only meaningful for multilabel classification where this differs from
              :func:`accuracy_score`).

            .. note::
                If `pos_label` is specified and `average` is None, then we report
                only the statistics of the positive class (i.e. equivalent to
                `average="binary"`).

        pos_label : int, float, bool or str, default=None
            The positive class.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The precision score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        >>> report.metrics.precision()
                                    LogisticRegression
                                                Split #0  Split #1
        Metric         Label / Average
        Precision     0                         0.96...   0.90...
                      1                         0.93...   0.96...
        """
        return self.report_metrics(
            scoring=["precision"],
            data_source=data_source,
            aggregate=aggregate,
            pos_label=pos_label,
            scoring_kwargs={"average": average},
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def recall(
        self,
        *,
        data_source: DataSource = "test",
        average: Optional[
            Literal["binary", "macro", "micro", "weighted", "samples"]
        ] = None,
        pos_label: Optional[Union[int, float, bool, str]] = None,
        aggregate: Optional[
            Union[Literal["mean", "std"], list[Literal["mean", "std"]]]
        ] = None,
    ) -> pd.DataFrame:
        """Compute the recall score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        average : {"binary","macro", "micro", "weighted", "samples"} or None, \
                default=None
            Used with multiclass problems.
            If `None`, the metrics for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            - "binary": Only report results for the class specified by `pos_label`.
              This is applicable only if targets (`y_{true,pred}`) are binary.
            - "micro": Calculate metrics globally by counting the total true positives,
              false negatives and false positives.
            - "macro": Calculate metrics for each label, and find their unweighted
              mean.  This does not take label imbalance into account.
            - "weighted": Calculate metrics for each label, and find their average
              weighted by support (the number of true instances for each label). This
              alters 'macro' to account for label imbalance; it can result in an F-score
              that is not between precision and recall. Weighted recall is equal to
              accuracy.
            - "samples": Calculate metrics for each instance, and find their average
              (only meaningful for multilabel classification where this differs from
              :func:`accuracy_score`).

            .. note::
                If `pos_label` is specified and `average` is None, then we report
                only the statistics of the positive class (i.e. equivalent to
                `average="binary"`).

        pos_label : int, float, bool or str, default=None
            The positive class.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The recall score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        >>> report.metrics.recall()
                                    LogisticRegression
                                            Split #0  Split #1
        Metric      Label / Average
        Recall     0                         0.87...  0.94...
                   1                         0.98...  0.94...
        """
        return self.report_metrics(
            scoring=["recall"],
            data_source=data_source,
            aggregate=aggregate,
            pos_label=pos_label,
            scoring_kwargs={"average": average},
        )

    @available_if(
        _check_supported_ml_task(supported_ml_tasks=["binary-classification"])
    )
    def brier_score(
        self,
        *,
        data_source: DataSource = "test",
        aggregate: Optional[
            Union[Literal["mean", "std"], list[Literal["mean", "std"]]]
        ] = None,
    ) -> pd.DataFrame:
        """Compute the Brier score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The Brier score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        >>> report.metrics.brier_score()
                        LogisticRegression
                                Split #0  Split #1
        Metric
        Brier score             0.04...   0.04...
        """
        return self.report_metrics(
            scoring=["brier_score"],
            data_source=data_source,
            aggregate=aggregate,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def roc_auc(
        self,
        *,
        data_source: DataSource = "test",
        average: Optional[Literal["macro", "micro", "weighted", "samples"]] = None,
        multi_class: Literal["raise", "ovr", "ovo"] = "ovr",
        aggregate: Optional[
            Union[Literal["mean", "std"], list[Literal["mean", "std"]]]
        ] = None,
    ) -> pd.DataFrame:
        """Compute the ROC AUC score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        average : {"macro", "micro", "weighted", "samples"}, default=None
            Average to compute the ROC AUC score in a multiclass setting. By default,
            no average is computed. Otherwise, this determines the type of averaging
            performed on the data.

            - "micro": Calculate metrics globally by considering each element of
              the label indicator matrix as a label.
            - "macro": Calculate metrics for each label, and find their unweighted
              mean. This does not take label imbalance into account.
            - "weighted": Calculate metrics for each label, and find their average,
              weighted by support (the number of true instances for each label).
            - "samples": Calculate metrics for each instance, and find their
              average.

            .. note::
                Multiclass ROC AUC currently only handles the "macro" and
                "weighted" averages. For multiclass targets, `average=None` is only
                implemented for `multi_class="ovr"` and `average="micro"` is only
                implemented for `multi_class="ovr"`.

        multi_class : {"raise", "ovr", "ovo"}, default="ovr"
            The multi-class strategy to use.

            - "raise": Raise an error if the data is multiclass.
            - "ovr": Stands for One-vs-rest. Computes the AUC of each class against the
              rest. This treats the multiclass case in the same way as the multilabel
              case. Sensitive to class imbalance even when `average == "macro"`,
              because class imbalance affects the composition of each of the "rest"
              groupings.
            - "ovo": Stands for One-vs-one. Computes the average AUC of all possible
              pairwise combinations of classes. Insensitive to class imbalance when
              `average == "macro"`.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The ROC AUC score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        >>> report.metrics.roc_auc()
                    LogisticRegression
                            Split #0  Split #1
        Metric
        ROC AUC             0.99...   0.98...
        """
        return self.report_metrics(
            scoring=["roc_auc"],
            data_source=data_source,
            aggregate=aggregate,
            scoring_kwargs={"average": average, "multi_class": multi_class},
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def log_loss(
        self,
        *,
        data_source: DataSource = "test",
        aggregate: Optional[
            Union[Literal["mean", "std"], list[Literal["mean", "std"]]]
        ] = None,
    ) -> pd.DataFrame:
        """Compute the log loss.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The log-loss.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        >>> report.metrics.log_loss()
                    LogisticRegression
                                Split #0  Split #1
        Metric
        Log loss                0.1...     0.1...
        """
        return self.report_metrics(
            scoring=["log_loss"],
            data_source=data_source,
            aggregate=aggregate,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["regression", "multioutput-regression"]
        )
    )
    def r2(
        self,
        *,
        data_source: DataSource = "test",
        multioutput: Literal["raw_values", "uniform_average"] = "raw_values",
        aggregate: Optional[
            Union[Literal["mean", "std"], list[Literal["mean", "std"]]]
        ] = None,
    ) -> pd.DataFrame:
        """Compute the R² score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        multioutput : {"raw_values", "uniform_average"} or array-like of shape \
                (n_outputs,), default="raw_values"
            Defines aggregating of multiple output values. Array-like value defines
            weights used to average errors. The other possible values are:

            - "raw_values": Returns a full set of errors in case of multioutput input.
            - "uniform_average": Errors of all outputs are averaged with uniform weight.

            By default, no averaging is done.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The R² score.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import CrossValidationReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> report = CrossValidationReport(regressor, X=X, y=y, cv_splitter=2)
        >>> report.metrics.r2()
                    Ridge
                Split #0  Split #1
        Metric
        R²      0.36...   0.39...
        """
        return self.report_metrics(
            scoring=["r2"],
            data_source=data_source,
            aggregate=aggregate,
            scoring_kwargs={"multioutput": multioutput},
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["regression", "multioutput-regression"]
        )
    )
    def rmse(
        self,
        *,
        data_source: DataSource = "test",
        multioutput: Literal["raw_values", "uniform_average"] = "raw_values",
        aggregate: Optional[
            Union[Literal["mean", "std"], list[Literal["mean", "std"]]]
        ] = None,
    ) -> pd.DataFrame:
        """Compute the root mean squared error.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        multioutput : {"raw_values", "uniform_average"} or array-like of shape \
                (n_outputs,), default="raw_values"
            Defines aggregating of multiple output values. Array-like value defines
            weights used to average errors. The other possible values are:

            - "raw_values": Returns a full set of errors in case of multioutput input.
            - "uniform_average": Errors of all outputs are averaged with uniform weight.

            By default, no averaging is done.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            The root mean squared error.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import CrossValidationReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> report = CrossValidationReport(regressor, X=X, y=y, cv_splitter=2)
        >>> report.metrics.rmse()
                    Ridge
                    Split #0   Split #1
        Metric
        RMSE        59.9...    61.4...
        """
        return self.report_metrics(
            scoring=["rmse"],
            data_source=data_source,
            aggregate=aggregate,
            scoring_kwargs={"multioutput": multioutput},
        )

    def custom_metric(
        self,
        metric_function: Callable,
        response_method: Union[str, list[str]],
        *,
        metric_name: Optional[str] = None,
        data_source: DataSource = "test",
        aggregate: Optional[
            Union[Literal["mean", "std"], list[Literal["mean", "std"]]]
        ] = None,
        **kwargs,
    ) -> pd.DataFrame:
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

        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the scores across the cross-validation splits.

        **kwargs : dict
            Any additional keyword arguments to be passed to the metric function.

        Returns
        -------
        pd.DataFrame
            The custom metric.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.metrics import mean_absolute_error
        >>> from skore import CrossValidationReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> report = CrossValidationReport(regressor, X=X, y=y, cv_splitter=2)
        >>> report.metrics.custom_metric(
        ...     metric_function=mean_absolute_error,
        ...     response_method="predict",
        ...     metric_name="MAE",
        ... )
                    Ridge
                Split #0   Split #1
        Metric
        MAE     50.1...   52.6...
        """
        # create a scorer with `greater_is_better=True` to not alter the output of
        # `metric_function`
        scorer = make_scorer(
            metric_function,
            greater_is_better=True,
            response_method=response_method,
            **kwargs,
        )
        return self.report_metrics(
            scoring=[scorer],
            data_source=data_source,
            aggregate=aggregate,
            scoring_names=[metric_name] if metric_name is not None else None,
        )

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def _sort_methods_for_help(
        self, methods: list[tuple[str, Callable]]
    ) -> list[tuple[str, Callable]]:
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

    def _format_method_name(self, name: str) -> str:
        """Override format method for metrics-specific naming."""
        method_name = f"{name}(...)"
        method_name = method_name.ljust(22)
        if name in self._SCORE_OR_LOSS_INFO and self._SCORE_OR_LOSS_INFO[name][
            "icon"
        ] in ("(↗︎)", "(↘︎)"):
            if self._SCORE_OR_LOSS_INFO[name]["icon"] == "(↗︎)":
                method_name += f"[cyan]{self._SCORE_OR_LOSS_INFO[name]['icon']}[/cyan]"
                return method_name.ljust(43)
            else:  # (↘︎)
                method_name += (
                    f"[orange1]{self._SCORE_OR_LOSS_INFO[name]['icon']}[/orange1]"
                )
                return method_name.ljust(49)
        else:
            return method_name.ljust(29)

    def _get_methods_for_help(self) -> list[tuple[str, Callable]]:
        """Override to exclude the plot accessor from methods list."""
        methods = super()._get_methods_for_help()
        return [(name, method) for name, method in methods if name != "plot"]

    def _get_help_panel_title(self) -> str:
        return "[bold cyan]Available metrics methods[/bold cyan]"

    def _get_help_legend(self) -> str:
        return (
            "[cyan](↗︎)[/cyan] higher is better [orange1](↘︎)[/orange1] lower is better"
        )

    def _get_help_tree_title(self) -> str:
        return "[bold cyan]report.metrics[/bold cyan]"

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(
            class_name="skore.CrossValidationReport.metrics",
            help_method_name="report.metrics.help()",
        )

    ####################################################################################
    # Methods related to displays
    ####################################################################################

    @progress_decorator(description="Computing predictions for display")
    def _get_display(
        self,
        *,
        data_source: DataSource,
        response_method: str,
        display_class: Any,
        display_kwargs: dict[str, Any],
    ) -> Union[RocCurveDisplay, PrecisionRecallCurveDisplay, PredictionErrorDisplay]:
        """Get the display from the cache or compute it.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        response_method : str
            The response method.

        display_class : class
            The display class.

        display_kwargs : dict
            The display kwargs used by `display_class._from_predictions`.

        Returns
        -------
        display : display_class
            The display.
        """
        if "random_state" in display_kwargs and display_kwargs["random_state"] is None:
            cache_key = None
        else:
            # Create a list of cache key components and then convert to tuple
            cache_key_parts: list[Any] = [self._parent._hash, display_class.__name__]
            cache_key_parts.extend(display_kwargs.values())
            cache_key_parts.append(data_source)
            cache_key = tuple(cache_key_parts)

        assert self._progress_info is not None, "Progress info not set"
        progress = self._progress_info["current_progress"]
        main_task = self._progress_info["current_task"]
        total_estimators = len(self._parent.estimator_reports_)
        progress.update(main_task, total=total_estimators)

        if cache_key in self._parent._cache:
            display = self._parent._cache[cache_key]
        else:
            y_true, y_pred = [], []
            for report in self._parent.estimator_reports_:
                X, y, _ = report.metrics._get_X_y_and_data_source_hash(
                    data_source=data_source
                )
                y_true.append(y)
                y_pred.append(
                    _get_cached_response_values(
                        cache=report._cache,
                        estimator_hash=report._hash,
                        estimator=report._estimator,
                        X=X,
                        response_method=response_method,
                        data_source=data_source,
                        data_source_hash=None,
                        pos_label=display_kwargs.get("pos_label"),
                    )
                )
                progress.update(main_task, advance=1, refresh=True)

            display = display_class._from_predictions(
                y_true,
                y_pred,
                estimator=self._parent.estimator_reports_[0]._estimator,
                estimator_name=self._parent.estimator_name_,
                ml_task=self._parent._ml_task,
                data_source=data_source,
                **display_kwargs,
            )

            # Unless random_state is an int (i.e. the call is deterministic),
            # we do not cache
            if cache_key is not None:
                self._parent._cache[cache_key] = display

        return display

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def roc(
        self,
        *,
        data_source: DataSource = "test",
        pos_label: Optional[Union[int, float, bool, str]] = None,
    ) -> RocCurveDisplay:
        """Plot the ROC curve.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        pos_label : int, float, bool or str, default=None
            The positive class.

        Returns
        -------
        RocCurveDisplay
            The ROC curve display.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        >>> display = report.metrics.roc()
        >>> display.plot(roc_curve_kwargs={"color": "tab:red"})
        """
        response_method = ("predict_proba", "decision_function")
        display_kwargs = {"pos_label": pos_label}
        display = self._get_display(
            data_source=data_source,
            response_method=response_method,
            display_class=RocCurveDisplay,
            display_kwargs=display_kwargs,
        )
        assert isinstance(display, RocCurveDisplay)
        return display

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def precision_recall(
        self,
        *,
        data_source: DataSource = "test",
        pos_label: Optional[Union[int, float, bool, str]] = None,
    ) -> PrecisionRecallCurveDisplay:
        """Plot the precision-recall curve.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        pos_label : int, float, bool or str, default=None
            The positive class.

        Returns
        -------
        PrecisionRecallCurveDisplay
            The precision-recall curve display.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        >>> display = report.metrics.precision_recall()
        >>> display.plot()
        """
        response_method = ("predict_proba", "decision_function")
        display_kwargs = {"pos_label": pos_label}
        display = self._get_display(
            data_source=data_source,
            response_method=response_method,
            display_class=PrecisionRecallCurveDisplay,
            display_kwargs=display_kwargs,
        )
        assert isinstance(display, PrecisionRecallCurveDisplay)
        return display

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["regression", "multioutput-regression"]
        )
    )
    def prediction_error(
        self,
        *,
        data_source: DataSource = "test",
        subsample: Union[float, int, None] = 1_000,
        random_state: Optional[int] = None,
    ) -> PredictionErrorDisplay:
        """Plot the prediction error of a regression model.

        Extra keyword arguments will be passed to matplotlib's `plot`.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        subsample : float, int or None, default=1_000
            Sampling the samples to be shown on the scatter plot. If `float`,
            it should be between 0 and 1 and represents the proportion of the
            original dataset. If `int`, it represents the number of samples
            display on the scatter plot. If `None`, no subsampling will be
            applied. by default, 1,000 samples or less will be displayed.

        random_state : int, default=None
            The random state to use for the subsampling.

        Returns
        -------
        PredictionErrorDisplay
            The prediction error display.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import CrossValidationReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> report = CrossValidationReport(regressor, X=X, y=y, cv_splitter=2)
        >>> display = report.metrics.prediction_error()
        >>> display.plot(kind="actual_vs_predicted", line_kwargs={"color": "tab:red"})
        """
        display_kwargs = {"subsample": subsample, "random_state": random_state}
        display = self._get_display(
            data_source=data_source,
            response_method="predict",
            display_class=PredictionErrorDisplay,
            display_kwargs=display_kwargs,
        )
        assert isinstance(display, PredictionErrorDisplay)
        return display
