from __future__ import annotations

import time
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, Optional

import joblib
import numpy as np

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseReport
from skore.sklearn._estimator.report import EstimatorReport

if TYPE_CHECKING:
    from skore.sklearn import EstimatorReport


def warn(title, message):
    from rich.panel import Panel

    from skore import console

    console.print(
        Panel(
            title=title,
            renderable=message,
            style="orange1",
            border_style="cyan",
        )
    )


class ComparisonReport(_BaseReport, DirNamesMixin):
    """Report for comparison of estimators.

    Parameters
    ----------
    reports : list of ``EstimatorReport``s
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

    See Also
    --------
    skore.sklearn.estimator.report.EstimatorReport
        Report for a fitted estimator.

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
        if len(reports) < 2:
            raise ValueError("At least 2 instances of EstimatorReport are needed")

        if not all(isinstance(report, EstimatorReport) for report in reports):
            raise TypeError("Only instances of EstimatorReport are allowed")

        if report_names is None:
            self.report_names_ = [report.estimator_name_ for report in reports]
        else:
            if len(report_names) != len(reports):
                raise ValueError(
                    "There should be as many report names as there are reports"
                )
            self.report_names_ = report_names

        self.estimator_reports_ = deepcopy(reports)

        # We check that the estimator reports can be compared:
        # - all estimators are in the same ML use case
        # - all X_train, y_train have the same hash (for estimator)
        # - all X_test, y_test have the same hash (for estimator)
        # - all reports are estimator reports (for now)

        def get_data_and_hash(report, source: Literal["train", "test"], /):
            X = getattr(report, f"X_{source}")
            y = getattr(report, f"y_{source}")

            return X, y, joblib.hash((X, y))

        first_report = self.estimator_reports_[0]
        first_ml_task = first_report._ml_task
        first_X_train, first_y_train, first_train_hash = get_data_and_hash(
            first_report, "train"
        )
        first_X_test, first_y_test, first_test_hash = get_data_and_hash(
            first_report, "test"
        )

        if first_X_train is None or first_y_train is None:
            warn(
                "MissingTrainingDataWarning",
                (
                    "We cannot ensure that all estimators have been trained "
                    "with the same dataset. This could lead to incoherent comparisons."
                ),
            )

        if first_X_test is None or first_y_test is None:
            warn(
                "MissingTestDataWarning",
                (
                    "We cannot ensure that all estimators have been tested "
                    "with the same dataset. This could lead to incoherent comparisons."
                ),
            )

        for report in self.estimator_reports_[1:]:
            if report._ml_task != first_ml_task:
                raise ValueError("Not all estimators are in the same ML usecase")

            _, _, train_hash = get_data_and_hash(report, "train")
            _, _, test_hash = get_data_and_hash(report, "test")

            if train_hash != first_train_hash:
                raise ValueError("Not all estimators have the same training data")

            if test_hash != first_test_hash:
                raise ValueError("Not all estimators have the same testing data")

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
