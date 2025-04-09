from __future__ import annotations

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseReport
from skore.sklearn._cross_validation.report import CrossValidationReport
from skore.utils._progress_bar import progress_decorator

if TYPE_CHECKING:
    from skore.sklearn._estimator.metrics_accessor import _MetricsAccessor


class CrossValidationComparisonReport(_BaseReport, DirNamesMixin):
    """Report for comparison of instances of :class:`skore.CrossValidationReport`.

    Caution: reports passed to `ComparisonReport` are not copied. If you pass
    a report to `ComparisonReport`, and then modify the report outside later, it will
    affect the report stored inside the `ComparisonReport` as well, which can lead to
    inconsistent results. For this reason, modifying reports after creation is strongly
    discouraged.

    Parameters
    ----------
    reports : list of :class:`~skore.CrossValidationReport` instances or dict
        Reports to compare.

        * If `reports` is a list, the class name of each estimator is used.
        * If `reports` is a dict, it is expected to have estimator names as keys
          and :class:`~skore.CrossValidationReport` instances as values.
          If the keys are not strings, they will be converted to strings.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        Some computations can be run in parallel, such as training the estimators
        and computing the scores.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors.

    Attributes
    ----------
    reports_ : list of :class:`~skore.CrossValidationReport`
        The compared reports.

    report_names_ : list of str
        The names of the compared reports.

    See Also
    --------
    skore.EstimatorReport
        Report for a fitted estimator.

    skore.CrossValidationReport
        Report for the cross-validation of an estimator.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import CrossValidationComparisonReport, CrossValidationReport
    >>> X, y = make_classification(random_state=42)
    >>> estimator_1 = LogisticRegression()
    >>> report_1 = CrossValidationReport(estimator_1, X, y)
    >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
    >>> report_2 = CrossValidationReport(estimator_2, X, y)
    >>> CrossValidationComparisonReport([report_1, report_2])
    ...
    >>> CrossValidationComparisonReport({"model1": report_1, "model2": report_2})
    ...
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]] = {
        "metrics": {"name": "metrics"},
    }
    metrics: _MetricsAccessor

    @staticmethod
    def _deduplicate_report_names(report_names_: list[str]) -> list[str]:
        """De-duplicate report names that appear several times.

        Leave the other report names alone.

        Examples
        --------
        >>> from skore import CrossValidationComparisonReport as CVCReport
        >>> CVCReport._deduplicate_report_names(['a', 'b'])
        ['a', 'b']
        >>> CVCReport._deduplicate_report_names(['a', 'a'])
        ['a_1', 'a_2']
        >>> CVCReport._deduplicate_report_names(['a', 'b', 'a'])
        ['a_1', 'b', 'a_2']
        >>> CVCReport._deduplicate_report_names(['a', 'b', 'a', 'b'])
        ['a_1', 'b_1', 'a_2', 'b_2']
        >>> CVCReport._deduplicate_report_names([])
        []
        >>> CVCReport._deduplicate_report_names(['a'])
        ['a']
        """
        result = report_names_.copy()
        for report_name in report_names_:
            indexes_of_report_names = [
                index for index, name in enumerate(report_names_) if name == report_name
            ]
            if len(indexes_of_report_names) == 1:
                # report name appears only once
                continue
            for n, index in enumerate(indexes_of_report_names, start=1):
                result[index] = f"{report_name}_{n}"
        return result

    @staticmethod
    def _validate_reports(
        reports: Union[list[CrossValidationReport], dict[str, CrossValidationReport]],
    ) -> tuple[list[CrossValidationReport], list[str]]:
        """Validate a list or dict of CrossValidationReports."""
        if not isinstance(reports, Iterable):
            raise TypeError(f"Expected reports to be an iterable; got {type(reports)}")

        if len(reports) < 2:
            raise ValueError("At least 2 instances of CrossValidationReport are needed")

        report_names = (
            list(map(str, reports.keys())) if isinstance(reports, dict) else None
        )
        reports = list(reports.values()) if isinstance(reports, dict) else reports

        if not all(isinstance(report, CrossValidationReport) for report in reports):
            raise TypeError("Expected instances of CrossValidationReport")

        ml_tasks = {report: report._ml_task for report in reports}
        if len(set(ml_tasks.values())) > 1:
            raise ValueError(
                f"Expected all estimators to have the same ML usecase; got {ml_tasks}"
            )

        if len(set(id(report) for report in reports)) < len(reports):
            raise ValueError("Compared CrossValidationReports must be distinct objects")

        if report_names is not None:
            report_names_ = report_names
        else:
            report_names_ = CrossValidationComparisonReport._deduplicate_report_names(
                [report.estimator_name_ for report in reports]
            )

        return reports, report_names_

    def __init__(
        self,
        reports: Union[list[CrossValidationReport], dict[str, CrossValidationReport]],
        *,
        n_jobs: Optional[int] = None,
    ) -> None:
        """
        ComparisonReport instance initializer.

        Notes
        -----
        We check that the reports can be compared:
        - all reports are :class:`~skore.CrossValidationReport`,
        - all estimators are in the same ML use case,
        """
        self.reports_, self.report_names_ = (
            CrossValidationComparisonReport._validate_reports(reports)
        )

        # used to know if a parent launches a progress bar manager
        self._progress_info: Optional[dict[str, Any]] = None
        self._parent_progress = None

        # NEEDED FOR METRICS ACCESSOR
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
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import ComparisonReport
        >>> X, y = make_classification(random_state=42)
        >>> estimator_1 = LogisticRegression()
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> report = CrossValidationComparisonReport([report_1, report_2])
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
        """Cache the predictions of the underlying estimator reports.

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
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import CrossValidationComparisonReport, CrossValidationReport
        >>> X, y = make_classification(random_state=42)
        >>> estimator_1 = LogisticRegression()
        >>> report_1 = CrossValidationReport(estimator_1, X, y)
        >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
        >>> report_2 = CrossValidationReport(estimator_2, X, y)
        >>> report = CrossValidationComparisonReport([report_1, report_2])
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
            # Pass the progress manager to child tasks
            report._parent_progress = progress
            report.cache_predictions(response_methods=response_methods, n_jobs=n_jobs)
            progress.update(main_task, advance=1, refresh=True)

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
