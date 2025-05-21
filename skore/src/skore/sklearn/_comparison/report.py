from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

import joblib
import numpy as np
from numpy.typing import ArrayLike

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseReport
from skore.sklearn._cross_validation.report import CrossValidationReport
from skore.sklearn._estimator.report import EstimatorReport
from skore.utils._progress_bar import progress_decorator

if TYPE_CHECKING:
    from skore.sklearn._estimator.metrics_accessor import _MetricsAccessor

    ReportType = Literal["EstimatorReport", "CrossValidationReport"]


class ComparisonReport(_BaseReport, DirNamesMixin):
    """Report for comparing reports.

    This object can be used to compare several :class:`skore.EstimatorReport` instances,
    or several :class:`~skore.CrossValidationReport` instances.

    .. caution:: Reports passed to `ComparisonReport` are not copied. If you pass
       a report to `ComparisonReport`, and then modify the report outside later, it
       will affect the report stored inside the `ComparisonReport` as well, which
       can lead to inconsistent results. For this reason, modifying reports after
       creation is strongly discouraged.

    Parameters
    ----------
    reports : list of reports or dict
        Reports to compare. If a dict, keys will be used to label the estimators;
        if a list, the labels are computed from the estimator class names.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimators and computing
        the scores are parallelized.
        When accessing some methods of the `ComparisonReport`, the `n_jobs`
        parameter is used to parallelize the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    reports_ : list of :class:`~skore.EstimatorReport` or list of
               :class:`~skore.CrossValidationReport`
        The compared reports.

    report_names_ : list of str
        The names of the compared estimators. If the names are not customized (i.e. the
        class names are used), a de-duplication process is used to make sure that the
        names are distinct.

    See Also
    --------
    skore.EstimatorReport
        Report for a fitted estimator.

    skore.CrossValidationReport
        Report for the cross-validation of an estimator.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from skore import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import ComparisonReport, EstimatorReport
    >>> X, y = make_classification(random_state=42)
    >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
    >>> estimator_1 = LogisticRegression()
    >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
    >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
    >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
    >>> report = ComparisonReport([estimator_report_1, estimator_report_2])
    >>> report.report_names_
    ['LogisticRegression_1', 'LogisticRegression_2']
    >>> report = ComparisonReport(
    ...     {"model1": estimator_report_1, "model2": estimator_report_2}
    ... )
    >>> report.report_names_
    ['model1', 'model2']

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import ComparisonReport, CrossValidationReport
    >>> X, y = make_classification(random_state=42)
    >>> estimator_1 = LogisticRegression()
    >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
    >>> report_1 = CrossValidationReport(estimator_1, X, y)
    >>> report_2 = CrossValidationReport(estimator_2, X, y)
    >>> report = ComparisonReport([report_1, report_2])
    >>> report = ComparisonReport({"model1": report_1, "model2": report_2})
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]] = {
        "metrics": {"name": "metrics"},
    }
    metrics: _MetricsAccessor

    _reports_type: ReportType

    @staticmethod
    def _validate_reports(
        reports: Union[
            list[EstimatorReport],
            dict[str, EstimatorReport],
            list[CrossValidationReport],
            dict[str, CrossValidationReport],
        ],
    ) -> tuple[
        Union[list[EstimatorReport], list[CrossValidationReport]],
        list[str],
        ReportType,
    ]:
        """Validate that reports are in the right format for comparison.

        Parameters
        ----------
        reports : list of reports or dict
            The reports to be validated.

        Returns
        -------
        list of EstimatorReport or list of CrossValidationReport
            The validated reports.
        list of str
            The report names, either taken from dict keys or computed from the estimator
            class names.
        {"EstimatorReport", "CrossValidationReport"}
            The inferred type of the reports that will be compared.
        """
        if not isinstance(reports, Iterable):
            raise TypeError(
                f"Expected reports to be a list or dict; got {type(reports)}"
            )

        if len(reports) < 2:
            raise ValueError(
                f"Expected at least 2 reports to compare; got {len(reports)}"
            )

        if isinstance(reports, list):
            report_names = None
            reports_list = reports
        else:  # dict
            report_names = list(reports.keys())
            for key in report_names:
                if not isinstance(key, str):
                    raise TypeError(
                        f"Expected all report names to be strings; got {type(key)}"
                    )
            reports_list = cast(
                Union[list[EstimatorReport], list[CrossValidationReport]],
                list(reports.values()),
            )

        reports_type: ReportType
        if all(isinstance(report, EstimatorReport) for report in reports_list):
            reports_list = cast(list[EstimatorReport], reports_list)
            reports_type = "EstimatorReport"

            test_dataset_hashes = {
                joblib.hash(report.y_test)
                for report in reports_list
                if report.y_test is not None
            }
            if len(test_dataset_hashes) > 1:
                raise ValueError(
                    "Expected all estimators to share the same test targets."
                )

        elif all(isinstance(report, CrossValidationReport) for report in reports_list):
            reports_list = cast(list[CrossValidationReport], reports_list)
            reports_type = "CrossValidationReport"
        else:
            raise TypeError(
                f"Expected list or dict of {EstimatorReport.__name__} "
                f"or list of dict of {CrossValidationReport.__name__}"
            )

        if len(set(id(report) for report in reports_list)) < len(reports_list):
            raise ValueError("Expected reports to be distinct objects")

        ml_tasks = {report: report._ml_task for report in reports_list}
        if len(set(ml_tasks.values())) > 1:
            raise ValueError(
                f"Expected all estimators to have the same ML usecase; got {ml_tasks}"
            )

        if report_names is None:
            deduped_report_names = _deduplicate_report_names(
                [report.estimator_name_ for report in reports_list]
            )
        else:
            deduped_report_names = report_names

        return reports_list, deduped_report_names, reports_type

    def __init__(
        self,
        reports: Union[
            list[EstimatorReport],
            dict[str, EstimatorReport],
            list[CrossValidationReport],
            dict[str, CrossValidationReport],
        ],
        *,
        n_jobs: Optional[int] = None,
    ) -> None:
        """
        ComparisonReport instance initializer.

        Notes
        -----
        We check that the estimator reports can be compared:
        - all reports are estimator reports,
        - all estimators are in the same ML use case,
        - all estimators have non-empty X_test and y_test,
        - all estimators have the same X_test and y_test.
        """
        self.reports_, self.report_names_, self._reports_type = (
            ComparisonReport._validate_reports(reports)
        )

        self._progress_info: Optional[dict[str, Any]] = None

        self.n_jobs = n_jobs
        self._rng = np.random.default_rng(time.time_ns())
        self._hash = self._rng.integers(
            low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max
        )
        self._cache: dict[tuple[Any, ...], Any] = {}
        self._ml_task = self.reports_[0]._ml_task

    def clear_cache(self) -> None:
        """Clear the cache.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = make_classification(random_state=42)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = LogisticRegression()
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> report = ComparisonReport([estimator_report_1, estimator_report_2])
        >>> report.cache_predictions()
        >>> report.clear_cache()
        >>> report._cache
        {}
        """
        for report in self.reports_:
            report.clear_cache()
        self._cache = {}

    @progress_decorator(description="Estimator predictions")
    def cache_predictions(
        self,
        response_methods: Literal[
            "auto", "predict", "predict_proba", "decision_function"
        ] = "auto",
        n_jobs: Optional[int] = None,
    ) -> None:
        """Cache the predictions for sub-estimators reports.

        Parameters
        ----------
        response_methods : {"auto", "predict", "predict_proba", "decision_function"},\
                default="auto
            The methods to use to compute the predictions.

        n_jobs : int, default=None
            The number of jobs to run in parallel. If `None`, we use the `n_jobs`
            parameter when initializing the report.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = make_classification(random_state=42)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = LogisticRegression()
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> report = ComparisonReport([estimator_report_1, estimator_report_2])
        >>> report.cache_predictions()
        >>> report._cache
        {...}
        """
        if n_jobs is None:
            n_jobs = self.n_jobs

        assert self._progress_info is not None, (
            "The rich Progress class was not initialized."
        )
        progress = self._progress_info["current_progress"]
        main_task = self._progress_info["current_task"]

        total_estimators = len(self.reports_)
        progress.update(main_task, total=total_estimators)

        for report in self.reports_:
            # Share the parent's progress bar with child report
            report._progress_info = {"current_progress": progress}
            report.cache_predictions(response_methods=response_methods, n_jobs=n_jobs)
            progress.update(main_task, advance=1, refresh=True)

    def get_predictions(
        self,
        *,
        data_source: Literal["train", "test", "X_y"],
        response_method: Literal["predict", "predict_proba", "decision_function"],
        X: Optional[ArrayLike] = None,
        pos_label: Optional[Any] = None,
    ) -> ArrayLike:
        """Get estimator's predictions.

        This method has the advantage to reload from the cache if the predictions
        were already computed in a previous call.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        response_method : {"predict", "predict_proba", "decision_function"}
            The response method to use.

        X : array-like of shape (n_samples, n_features), optional
            When `data_source` is "X_y", the input features on which to compute the
            response method.

        pos_label : int, float, bool or str, default=None
            The positive class when it comes to binary classification. When
            `response_method="predict_proba"`, it will select the column corresponding
            to the positive class. When `response_method="decision_function"`, it will
            negate the decision function if `pos_label` is different from
            `estimator.classes_[1]`.

        Returns
        -------
        list of np.ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predictions for each cross-validation split.

        Raises
        ------
        ValueError
            If the data source is invalid.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from skore import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = make_classification(random_state=42)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = LogisticRegression()
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> report = ComparisonReport([estimator_report_1, estimator_report_2])
        >>> report.cache_predictions()
        >>> predictions = report.get_predictions(
        ...     data_source="test", response_method="predict"
        ... )
        >>> print([split_predictions.shape for split_predictions in predictions])
        [(25,), (25,)]
        """
        return [
            report.get_predictions(
                data_source=data_source,
                response_method=response_method,
                X=X,
                pos_label=pos_label,
            )
            for report in self.reports_
        ]

    ####################################################################################
    # Methods related to the help and repr
    ####################################################################################

    def _get_help_panel_title(self) -> str:
        return "[bold cyan]Tools to compare estimators[/bold cyan]"

    def _get_help_legend(self) -> str:
        return (
            "[cyan](↗︎)[/cyan] higher is better [orange1](↘︎)[/orange1] lower is better"
        )

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{self.__class__.__name__}(...)"


def _deduplicate_report_names(report_names: list[str]) -> list[str]:
    """De-duplicate report names that appear several times.

    Leave the other report names alone.

    Parameters
    ----------
    report_names : list of str
        The list of report names to be checked.

    Returns
    -------
    list of str
        The de-duplicated list of report names.

    Examples
    --------
    >>> _deduplicate_report_names(['a', 'b'])
    ['a', 'b']
    >>> _deduplicate_report_names(['a', 'a'])
    ['a_1', 'a_2']
    >>> _deduplicate_report_names(['a', 'b', 'a'])
    ['a_1', 'b', 'a_2']
    >>> _deduplicate_report_names(['a', 'b', 'a', 'b'])
    ['a_1', 'b_1', 'a_2', 'b_2']
    >>> _deduplicate_report_names([])
    []
    >>> _deduplicate_report_names(['a'])
    ['a']
    """
    counts = Counter(report_names)
    if len(report_names) == len(counts):
        return report_names

    names = report_names.copy()
    seen: Counter = Counter()
    for i in range(len(names)):
        name = names[i]
        seen[name] += 1
        if counts[name] > 1:
            names[i] = f"{name}_{seen[name]}"
    return names
