import copy
from typing import Any, Callable, Literal, Optional, Union

import joblib
import numpy as np
import pandas as pd
from rich.progress import Progress
from sklearn.metrics import make_scorer
from sklearn.metrics._scorer import _BaseScorer as SKLearnScorer
from sklearn.utils.metaestimators import available_if

from skore import config_context
from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseAccessor
from skore.sklearn._plot.metrics import (
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from skore.sklearn.types import Aggregate
from skore.utils._accessor import _check_supported_ml_task
from skore.utils._fixes import _validate_joblib_parallel_params
from skore.utils._index import flatten_multi_index
from skore.utils._progress_bar import progress_decorator

from .report import CrossValidationComparisonReport

DataSource = Literal["test", "train"]


class _MetricsAccessor(_BaseAccessor, DirNamesMixin):
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

    def __init__(self, parent: CrossValidationComparisonReport) -> None:
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
        aggregate: Optional[Aggregate] = ("mean", "std"),
    ) -> pd.DataFrame:
        """Report a set of metrics for the estimators.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        scoring : list of str, callable, or scorer, default=None
            The metrics to report. You can get the possible list of strings by calling
            `report.metrics.help()`. When passing a callable, it should take as
            arguments ``y_true``, ``y_pred`` as the two first arguments. Additional
            arguments can be passed as keyword arguments and will be forwarded with
            `scoring_kwargs`. If the callable API is too restrictive (e.g. need to pass
            same parameter name with different values), you can use scikit-learn scorers
            as provided by :func:`sklearn.metrics.make_scorer`.

        scoring_names : list of str, default=None
            Used to overwrite the default scoring names in the report. It should be of
            the same length as the ``scoring`` parameter.

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

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        Returns
        -------
        pd.DataFrame
            The statistics for the metrics.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import CrossValidationComparisonReport, CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = DummyClassifier()
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = DummyClassifier(strategy="most_frequent")
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> comparison_report = CrossValidationComparisonReport([report_1, report_2])
        >>> comparison_report.metrics.report_metrics(
        ...     scoring=["precision", "recall"],
        ...     pos_label=1,
        ... )
        >>> comparison_report.metrics.report_metrics(
        ...     scoring=["precision", "recall"],
        ...     pos_label=1,
        ...     aggregate=None,
        ... )
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

    @staticmethod
    def _combine_cross_validation_results(
        results: list[pd.DataFrame],
        estimator_names: list[str],
        indicator_favorability: bool,
        aggregate: Optional[Aggregate],
    ) -> pd.DataFrame:
        """Combine a list of dataframes.

        Parameters
        ----------
        results : pd.DataFrame
            The dataframes to combine.
            They are assumed to originate from a `CrossValidationReport.metrics`
            computation. In particular, there are several assumptions made:

            - every dataframe has the form:
                - index: Index "Metric", or MultiIndex ["Metric", "Label / Average"]
                - columns: MultiIndex with levels ["Estimator", "Splits"] (can be
                  unnamed)
            - all dataframes have the same metrics

            The dataframes are not required to have the same number of columns (splits).

        estimator_names : list of str of len (len(results))
            The name to give the estimator for each dataframe.

        indicator_favorability : bool
            Whether to keep the Favorability column.

        aggregate : Aggregate
            How to aggregate the resulting dataframe.
        """

        def add_model_name_to_index(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
            """Move the model name from the column index to the table index."""
            df = copy.copy(df)

            # Put the model name as a column
            df["Estimator"] = model_name

            # Put the model name into the index
            if "Label / Average" in df.index.names:
                new_index = ["Metric", "Label / Average", "Estimator"]
            else:
                new_index = ["Metric", "Estimator"]
            df = df.reset_index().set_index(new_index)

            # Then drop the model from the columns
            df.columns = df.columns.droplevel(0)

            return df

        def reshape_results(df):
            """Put data in the correct format.

            - Index is "Metric", optionally "Label / Average", "Split"
            - Columns are "Estimator"

            Examples
            --------
            >>> # xdoctest: +SKIP
            >>> df
                                                    Split #0  Split #1  Split #2
            Estimator Metric       Label / Average
            m1        Precision    0                1.000000  1.000000  1.000000
                                   1                1.000000  1.000000  1.000000
                      ROC AUC                       1.000000  1.000000  1.000000
            m2        Precision    0                1.000000  1.000000       NaN
                                   1                1.000000  0.941176       NaN
                      ROC AUC                       1.000000  0.996324       NaN
            >>> reshape_results(df)
            Estimator                                   m1        m2
            Metric      Label / Average Split
            Precision   0               Split #0  1.000000  1.000000
                                        Split #1  1.000000  1.000000
                                        Split #2  1.000000       NaN
                        1               Split #0  1.000000  1.000000
                                        Split #1  1.000000  0.941176
                                        Split #2  1.000000       NaN
            ROC AUC                     Split #0  1.000000  1.000000
                                        Split #1  1.000000  0.996324
                                        Split #2  1.000000       NaN
            """
            splits = df.columns

            df_reset = df.reset_index()

            metric_order = df_reset["Metric"].unique()

            # Melt the Split columns into rows
            if "Label / Average" in df_reset.columns:
                id_vars = ["Estimator", "Metric", "Label / Average"]
            else:
                id_vars = ["Estimator", "Metric"]
            melted_df = pd.melt(
                df_reset,
                id_vars=id_vars,
                value_vars=splits,
                var_name="Split",
                value_name="Value",
            )

            # Now pivot to have models as columns
            if "Label / Average" in melted_df.columns:
                index = ["Metric", "Label / Average", "Split"]
            else:
                index = ["Metric", "Split"]
            result_df = pd.pivot_table(
                melted_df,
                index=index,
                columns="Estimator",
                values="Value",
            )

            result_df = result_df.reindex(metric_order, level="Metric")

            return result_df

        results = results.copy()

        # Pop the favorability column if it exists, to:
        # - not use it in the aggregate operation
        # - later to only report a single column and not by split columns
        if indicator_favorability:
            favorability = results[0]["Favorability"]
            for result in results:
                result.pop("Favorability")
        else:
            favorability = None

        df_model_name_in_index = pd.concat(
            [
                add_model_name_to_index(df, estimator_name)
                for df, estimator_name in zip(results, estimator_names)
            ]
        )

        if aggregate:
            if isinstance(aggregate, str):
                aggregate = [aggregate]

            df = df_model_name_in_index.aggregate(func=aggregate, axis=1)
        else:
            df = reshape_results(df_model_name_in_index)

        if favorability is not None:
            df["Favorability"] = favorability

        return df

    @progress_decorator(description="Compute metric for each CrossValidationReport")
    def _compute_metric_scores(
        self,
        report_metric_name: str,
        *,
        data_source: DataSource = "test",
        aggregate: Optional[
            Union[Literal["mean", "std"], list[Literal["mean", "std"]]]
        ] = None,
        **metric_kwargs: Any,
    ):
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

        # we need to enforce the order of the parameter for a specific metric
        # to make sure that we hit the cache in a consistent way
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

        total_estimators = len(self._parent.reports_)
        progress.update(main_task, total=total_estimators)

        if cache_key in self._parent._cache:
            results = self._parent._cache[cache_key]
        else:
            parallel = joblib.Parallel(
                **_validate_joblib_parallel_params(
                    n_jobs=self._parent.n_jobs,
                    return_as="generator",
                    require="sharedmem",
                )
            )
            generator = parallel(
                joblib.delayed(getattr(report.metrics, report_metric_name))(
                    data_source=data_source,
                    aggregate=None,
                    **metric_kwargs,
                )
                for report in self._parent.reports_
            )
            individual_results = []
            with config_context(show_progress=False):
                for result in generator:
                    individual_results.append(result)
                    progress.update(main_task, advance=1, refresh=True)

            results = _MetricsAccessor._combine_cross_validation_results(
                individual_results,
                self._parent.report_names_,
                indicator_favorability=metric_kwargs.get(
                    "indicator_favorability", False
                ),
                aggregate=aggregate,
            )

            self._parent._cache[cache_key] = results
        return results

    def timings(
        self,
        aggregate: Optional[Aggregate] = ("mean", "std"),
    ) -> pd.DataFrame:
        """Get all measured processing times.

        The index of the returned dataframe is the name of the processing time. When
        the estimators were not used to predict, no timings regarding the prediction
        will be present.

        Parameters
        ----------
        aggregate : {"mean", "std"} or list of such str, default=None
            Function to aggregate the timings across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            A dataframe with the processing times.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import CrossValidationComparisonReport, CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = DummyClassifier()
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = DummyClassifier(strategy="most_frequent")
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> comparison_report = CrossValidationComparisonReport([report_1, report_2])
        >>> comparison_report.cache_predictions(response_methods=["predict"])
        >>> comparison_report.metrics.timings()
        """
        results = [
            report.metrics.timings(aggregate=None) for report in self._parent.reports_
        ]

        # Put dataframes in the right shape
        for i, result in enumerate(results):
            result.index.name = "Metric"
            result.columns = pd.MultiIndex.from_product(
                [[self._parent.report_names_[i]], result.columns]
            )

        timings = self._combine_cross_validation_results(
            results,
            self._parent.report_names_,
            indicator_favorability=False,
            aggregate=aggregate,
        )
        return timings

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def accuracy(
        self,
        *,
        data_source: DataSource = "test",
        aggregate: Optional[Aggregate] = ("mean", "std"),
    ) -> pd.DataFrame:
        """Compute the accuracy score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        Returns
        -------
        pd.DataFrame
            The accuracy score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import CrossValidationComparisonReport, CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = DummyClassifier()
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = DummyClassifier(strategy="most_frequent")
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> comparison_report = CrossValidationComparisonReport([report_1, report_2])
        >>> comparison_report.metrics.accuracy()
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
        aggregate: Optional[Aggregate] = ("mean", "std"),
    ) -> pd.DataFrame:
        """Compute the precision score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        average : {"binary", "macro", "micro", "weighted", "samples"} or None, \
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

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        Returns
        -------
        pd.DataFrame
            The precision score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import CrossValidationComparisonReport, CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = DummyClassifier(strategy="uniform", random_state=0)
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = DummyClassifier(strategy="uniform", random_state=1)
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> comparison_report = CrossValidationComparisonReport([report_1, report_2])
        >>> comparison_report.metrics.precision()
        """
        return self.report_metrics(
            scoring=["precision"],
            data_source=data_source,
            pos_label=pos_label,
            scoring_kwargs={"average": average},
            aggregate=aggregate,
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
        aggregate: Optional[Aggregate] = ("mean", "std"),
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

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        Returns
        -------
        pd.DataFrame
            The recall score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import CrossValidationComparisonReport, CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = DummyClassifier()
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = DummyClassifier(strategy="most_frequent")
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> comparison_report = CrossValidationComparisonReport([report_1, report_2])
        >>> comparison_report.metrics.recall()
        """
        return self.report_metrics(
            scoring=["recall"],
            data_source=data_source,
            pos_label=pos_label,
            scoring_kwargs={"average": average},
            aggregate=aggregate,
        )

    @available_if(
        _check_supported_ml_task(supported_ml_tasks=["binary-classification"])
    )
    def brier_score(
        self,
        *,
        data_source: DataSource = "test",
        aggregate: Optional[Aggregate] = ("mean", "std"),
    ) -> pd.DataFrame:
        """Compute the Brier score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        Returns
        -------
        pd.DataFrame
            The Brier score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import CrossValidationComparisonReport, CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = DummyClassifier()
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = DummyClassifier(strategy="most_frequent")
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> comparison_report = CrossValidationComparisonReport([report_1, report_2])
        >>> comparison_report.metrics.brier_score()
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
        average: Optional[
            Literal["auto", "macro", "micro", "weighted", "samples"]
        ] = None,
        multi_class: Literal["raise", "ovr", "ovo"] = "ovr",
        aggregate: Optional[Aggregate] = ("mean", "std"),
    ) -> pd.DataFrame:
        """Compute the ROC AUC score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        average : {"auto", "macro", "micro", "weighted", "samples"}, \
                default=None
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

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        Returns
        -------
        pd.DataFrame
            The ROC AUC score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import CrossValidationComparisonReport, CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = DummyClassifier()
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = DummyClassifier(strategy="most_frequent")
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> comparison_report = CrossValidationComparisonReport([report_1, report_2])
        >>> comparison_report.metrics.roc_auc()
        """
        return self.report_metrics(
            scoring=["roc_auc"],
            data_source=data_source,
            scoring_kwargs={"average": average, "multi_class": multi_class},
            aggregate=aggregate,
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
        aggregate: Optional[Aggregate] = ("mean", "std"),
    ) -> pd.DataFrame:
        """Compute the log loss.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        Returns
        -------
        pd.DataFrame
            The log-loss.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import CrossValidationComparisonReport, CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = DummyClassifier()
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = DummyClassifier(strategy="most_frequent")
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> comparison_report = CrossValidationComparisonReport([report_1, report_2])
        >>> comparison_report.metrics.log_loss()
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
        aggregate: Optional[Aggregate] = ("mean", "std"),
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

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        Returns
        -------
        pd.DataFrame
            The R² score.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.dummy import DummyRegressor
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import CrossValidationComparisonReport, CrossValidationReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> estimator_1 = DummyRegressor()
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = DummyRegressor(strategy="median")
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> comparison_report = CrossValidationComparisonReport([report_1, report_2])
        >>> comparison_report.metrics.r2()
        """
        return self.report_metrics(
            scoring=["r2"],
            data_source=data_source,
            scoring_kwargs={"multioutput": multioutput},
            aggregate=aggregate,
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
        aggregate: Optional[Aggregate] = ("mean", "std"),
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

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        Returns
        -------
        pd.DataFrame
            The root mean squared error.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.dummy import DummyRegressor
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import CrossValidationComparisonReport, CrossValidationReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> estimator_1 = DummyRegressor()
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = DummyRegressor(strategy="median")
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> comparison_report = CrossValidationComparisonReport([report_1, report_2])
        >>> comparison_report.metrics.rmse()
        """
        return self.report_metrics(
            scoring=["rmse"],
            data_source=data_source,
            scoring_kwargs={"multioutput": multioutput},
            aggregate=aggregate,
        )

    def custom_metric(
        self,
        metric_function: Callable,
        response_method: Union[str, list[str]],
        *,
        metric_name: Optional[str] = None,
        data_source: DataSource = "test",
        aggregate: Optional[Aggregate] = ("mean", "std"),
        **kwargs: Any,
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

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        **kwargs : dict
            Any additional keyword arguments to be passed to the metric function.

        Returns
        -------
        pd.DataFrame
            The custom metric.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.dummy import DummyRegressor
        >>> from sklearn.metrics import mean_absolute_error
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import CrossValidationComparisonReport, CrossValidationReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> estimator_1 = DummyRegressor()
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = DummyRegressor(strategy="median")
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> comparison_report = CrossValidationComparisonReport([report_1, report_2])
        >>> comparison_report.metrics.custom_metric(
        ...     metric_function=mean_absolute_error,
        ...     response_method="predict",
        ...     metric_name="MAE",
        ... )
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
            scoring_names=[metric_name] if metric_name is not None else None,
            aggregate=aggregate,
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
            class_name="skore.CrossValidationComparisonReport.metrics",
        )

    ####################################################################################
    # Methods related to displays
    ####################################################################################

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
        >>> # xdoctest: +SKIP
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import CrossValidationComparisonReport, CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = DummyClassifier()
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = DummyClassifier(strategy="most_frequent")
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> comparison_report = CrossValidationComparisonReport([report_1, report_2])
        >>> display = comparison_report.metrics.roc()
        >>> display.plot()
        """
        raise NotImplementedError()

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
        data_source : {"test", "train"}, default="test"
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
        >>> # xdoctest: +SKIP
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import CrossValidationComparisonReport, CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = DummyClassifier()
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = DummyClassifier(strategy="most_frequent")
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> comparison_report = CrossValidationComparisonReport([report_1, report_2])
        >>> display = comparison_report.metrics.precision_recall()
        >>> display.plot()
        """
        raise NotImplementedError()

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["regression", "multioutput-regression"]
        )
    )
    def prediction_error(
        self,
        *,
        data_source: DataSource = "test",
        subsample: int = 1_000,
        seed: Optional[int] = None,
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

        seed : int, default=None
            The seed used to initialize the random number generator used for the
            subsampling.

        Returns
        -------
        PredictionErrorDisplay
            The prediction error display.

        Examples
        --------
        >>> # xdoctest: +SKIP
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.dummy import DummyRegressor
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import CrossValidationComparisonReport, CrossValidationReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> estimator_1 = DummyRegressor()
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = DummyRegressor(strategy="median")
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> comparison_report = CrossValidationComparisonReport([report_1, report_2])
        >>> display = comparison_report.metrics.prediction_error()
        >>> display.plot(kind="actual_vs_predicted")
        """
        raise NotImplementedError()
