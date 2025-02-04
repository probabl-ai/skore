from __future__ import annotations

import time
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

import joblib
import numpy as np

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn import EstimatorReport
from skore.sklearn._base import _BaseReport

if TYPE_CHECKING:
    from skore.sklearn import EstimatorReport


class Comparator(_BaseReport, DirNamesMixin):
    """"""

    _ACCESSOR_CONFIG = {
        "metrics": {"name": "metrics"},
    }

    def __init__(
        self,
        estimator_reports: list[EstimatorReport],
        n_jobs=None,
    ):
        if len(estimator_reports) < 2:
            raise ValueError("At least 2 instances of EstimatorReport are needed")

        self.estimator_reports_ = deepcopy(estimator_reports)
        self._ml_task = self.estimator_reports_[0]._ml_task

        # We check that the estimator reports can be compared:
        # - all estimators are in the same ml use case
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

        if not isinstance(first_report, EstimatorReport):
            raise ValueError("Only instances of EstimatorReport are allowed")

        if first_X_train is None or first_y_train is None:
            warnings.warn(
                "Not all estimators have training data, this could lead to incoherent comparisons",
                stacklevel=1,
            )

        if first_X_test is None or first_y_test is None:
            warnings.warn(
                "Not all estimators have testing data, this could lead to incoherent comparisons",
                stacklevel=1,
            )

        for report in self.estimator_reports_[1:]:
            if not isinstance(report, EstimatorReport):
                raise ValueError("Only instances of EstimatorReport are allowed")

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
        return self._rich_repr(class_name="skore.Comparator", help_method_name="help()")
