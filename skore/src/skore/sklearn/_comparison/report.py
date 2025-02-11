from __future__ import annotations

import time
from collections.abc import Iterable
from copy import deepcopy
from typing import Optional

import joblib
import numpy as np

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseReport
from skore.sklearn._estimator.report import EstimatorReport


class ComparisonReport(_BaseReport, DirNamesMixin):
    """Report for comparison of :class:`skore.EstimatorReport`.

    Parameters
    ----------
    reports : list of :class:`skore.EstimatorReport`s
        Estimator reports to compare.

    report_names : list of str, default=None
        Used to name the compared reports. It should be of
        the same length as the `reports` parameter.
        If None, each report is named after its estimator's class.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimators and computing
        the scores are parallelized.
        When accessing some methods of the `ComparisonReport`, the `n_jobs`
        parameter is used to parallelize the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    estimator_reports_ : list of EstimatorReport
        The estimator reports for each split.

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
    ...
    """

    _ACCESSOR_CONFIG = {
        "metrics": {"name": "metrics"},
    }

    def __init__(
        self,
        reports: list[EstimatorReport],
        *,
        report_names: Optional[list[str]] = None,
        n_jobs: Optional[int] = None,
    ):
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
            raise ValueError("At least 2 instances of EstimatorReport are needed")

        if not all(isinstance(report, EstimatorReport) for report in reports):
            raise TypeError("Expected instances of EstimatorReport")

        for report in reports:
            if (report.X_test is None) or (report.y_test is None):
                raise ValueError("Cannot compare reports without testing data")

        test_dataset_hashes = {
            report: joblib.hash((report.X_test, report.y_test)) for report in reports
        }
        if len(set(test_dataset_hashes.values())) > 1:
            raise ValueError(
                "Expected all estimators to have the same testing data; "
                f"got {test_dataset_hashes}"
            )

        ml_tasks = {report: report._ml_task for report in reports}
        if len(set(ml_tasks.values())) > 1:
            raise ValueError(
                f"Expected all estimators to have the same ML usecase; "
                f"got {ml_tasks}"
            )

        if report_names is None:
            self.report_names_ = [report.estimator_name_ for report in reports]
        elif len(report_names) == len(reports):
            self.report_names_ = report_names
        else:
            raise ValueError(
                "Expected as many report names as there are reports; "
                f"got {len(report_names)} report names but {len(reports)} reports"
            )

        self.estimator_reports_ = deepcopy(reports)

        # NEEDED FOR METRICS ACCESSOR
        self.n_jobs = n_jobs
        self._rng = np.random.default_rng(time.time_ns())
        self._hash = self._rng.integers(
            low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max
        )
        self._cache = {}
        self._ml_task = self.estimator_reports_[0]._ml_task

    ####################################################################################
    # Methods related to the help and repr
    ####################################################################################

    def _get_help_panel_title(self):
        return "[bold cyan]Tools to compare estimators[/bold cyan]"

    def _get_help_legend(self):
        return (
            "[cyan](↗︎)[/cyan] higher is better [orange1](↘︎)[/orange1] lower is better"
        )

    def __repr__(self):
        """Return a string representation using rich."""
        return self._rich_repr(
            class_name="skore.ComparisonReport", help_method_name="help()"
        )
