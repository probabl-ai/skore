from __future__ import annotations

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

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


class ComparisonReport(_BaseReport, DirNamesMixin):
    """Report for comparison of instances of :class:`skore.EstimatorReport`.

    .. caution:: Reports passed to `ComparisonReport` are not copied. If you pass
       a report to `ComparisonReport`, and then modify the report outside later, it
       will affect the report stored inside the `ComparisonReport` as well, which
       can lead to inconsistent results. For this reason, modifying reports after
       creation is strongly discouraged.

    Parameters
    ----------
    reports : list of :class:`~skore.EstimatorReport` instances or dict
        Estimator reports to compare.

        * If `reports` is a list, the class name of each estimator is used.
        * If `reports` is a dict, it is expected to have estimator names as keys
          and :class:`~skore.EstimatorReport` instances as values.
          If the keys are not strings, they will be converted to strings.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimators and computing
        the scores are parallelized.
        When accessing some methods of the `ComparisonReport`, the `n_jobs`
        parameter is used to parallelize the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    reports_ : list of :class:`~skore.EstimatorReport`
        The compared estimator reports.

    report_names_ : list of str
        The names of the compared estimator reports.

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
    >>> from skore import ComparisonReport, EstimatorReport
    >>> X, y = make_classification(random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> estimator_1 = LogisticRegression()
    >>> estimator_report_1 = EstimatorReport(
    ...     estimator_1,
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_test=X_test,
    ...     y_test=y_test
    ... )
    >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
    >>> estimator_report_2 = EstimatorReport(
    ...     estimator_2,
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_test=X_test,
    ...     y_test=y_test
    ... )
    >>> report = ComparisonReport([estimator_report_1, estimator_report_2])
    >>> report = ComparisonReport(
    ...     {"model1": estimator_report_1, "model2": estimator_report_2}
    ... )
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
        >>> ComparisonReport._deduplicate_report_names(['a', 'b'])
        ['a', 'b']
        >>> ComparisonReport._deduplicate_report_names(['a', 'a'])
        ['a_1', 'a_2']
        >>> ComparisonReport._deduplicate_report_names(['a', 'b', 'a'])
        ['a_1', 'b', 'a_2']
        >>> ComparisonReport._deduplicate_report_names(['a', 'b', 'a', 'b'])
        ['a_1', 'b_1', 'a_2', 'b_2']
        >>> ComparisonReport._deduplicate_report_names([])
        []
        >>> ComparisonReport._deduplicate_report_names(['a'])
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
    def _validate_cross_validation_reports(
        reports: list[Any], report_names: Optional[list[str]]
    ) -> tuple[list[CrossValidationReport], list[str]]:
        """Validate CrossValidationReports."""
        if not all(isinstance(report, CrossValidationReport) for report in reports):
            raise TypeError("Expected instances of CrossValidationReport")

        if len(set(id(report) for report in reports)) < len(reports):
            raise ValueError("Compared CrossValidationReports must be distinct objects")

        if report_names is not None:
            report_names_ = report_names
        else:
            report_names_ = ComparisonReport._deduplicate_report_names(
                [report.estimator_name_ for report in reports]
            )

        return reports, report_names_

    @staticmethod
    def _validate_estimator_reports(
        reports: list[Any], report_names: Optional[list[str]]
    ) -> tuple[list[EstimatorReport], list[str]]:
        """Validate EstimatorReports."""
        if not all(isinstance(report, EstimatorReport) for report in reports):
            raise TypeError("Expected instances of EstimatorReport")

        test_dataset_hashes = {
            joblib.hash((report.X_test, report.y_test))
            for report in reports
            if not ((report.X_test is None) and (report.y_test is None))
        }
        if len(test_dataset_hashes) > 1:
            raise ValueError("Expected all estimators to have the same testing data.")

        if report_names is None:
            report_names_ = [report.estimator_name_ for report in reports]
        else:
            report_names_ = report_names

        return reports, report_names_

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
        if not isinstance(reports, Iterable):
            raise TypeError(f"Expected reports to be an iterable; got {type(reports)}")

        if len(reports) < 2:
            raise ValueError("Expected at least 2 reports to compare")

        report_names = (
            list(map(str, reports.keys())) if isinstance(reports, dict) else None
        )
        reports_list = list(reports.values()) if isinstance(reports, dict) else reports

        if isinstance(reports_list[0], EstimatorReport):
            self.reports_, self.report_names_ = (
                ComparisonReport._validate_estimator_reports(
                    reports_list,
                    report_names,
                )
            )
        elif isinstance(reports_list[0], CrossValidationReport):
            self.reports_, self.report_names_ = (
                ComparisonReport._validate_cross_validation_reports(
                    reports_list,
                    report_names,
                )
            )
        else:
            raise TypeError(
                f"Expected instances of {EstimatorReport.__name__} "
                f"or {CrossValidationReport.__name__}, "
                f"got {type(reports_list[0])}"
            )

        ml_tasks = {report: report._ml_task for report in self.reports_}
        if len(set(ml_tasks.values())) > 1:
            raise ValueError(
                f"Expected all estimators to have the same ML usecase; got {ml_tasks}"
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
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        >>> estimator_1 = LogisticRegression()
        >>> estimator_report_1 = EstimatorReport(
        ...     estimator_1,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test
        ... )
        >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
        >>> estimator_report_2 = EstimatorReport(
        ...     estimator_2,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test
        ... )
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
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = make_classification(random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        >>> estimator_1 = LogisticRegression()
        >>> estimator_report_1 = EstimatorReport(
        ...     estimator_1,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test
        ... )
        >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
        >>> estimator_report_2 = EstimatorReport(
        ...     estimator_2,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test
        ... )
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
            # Pass the progress manager to child tasks
            report._parent_progress = progress
            report.cache_predictions(
                response_methods=response_methods, n_jobs=n_jobs
            )
            progress.update(main_task, advance=1, refresh=True)

    def get_predictions(
        self,
        *,
        data_source: Literal["train", "test", "X_y"],
        response_method: Literal["predict", "predict_proba", "decision_function"],
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
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = make_classification(random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        >>> estimator_1 = LogisticRegression()
        >>> estimator_report_1 = EstimatorReport(
        ...     estimator_1,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test
        ... )
        >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
        >>> estimator_report_2 = EstimatorReport(
        ...     estimator_2,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test
        ... )
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
