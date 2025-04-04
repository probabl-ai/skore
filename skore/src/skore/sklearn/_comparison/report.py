from __future__ import annotations

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseReport
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
    estimator_reports_ : list of :class:`~skore.EstimatorReport`
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
    ...
    >>> report = ComparisonReport(
    ...     {"model1": estimator_report_1, "model2": estimator_report_2}
    ... )
    ...
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]] = {
        "metrics": {"name": "metrics"},
    }
    metrics: _MetricsAccessor

    def __init__(
        self,
        reports: Union[list[EstimatorReport], dict[str, EstimatorReport]],
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
            raise ValueError("At least 2 instances of EstimatorReport are needed")

        report_names = (
            list(map(str, reports.keys())) if isinstance(reports, dict) else None
        )
        reports = list(reports.values()) if isinstance(reports, dict) else reports

        if not all(isinstance(report, EstimatorReport) for report in reports):
            raise TypeError("Expected instances of EstimatorReport")

        test_dataset_hashes = {
            joblib.hash((report.X_test, report.y_test))
            for report in reports
            if not ((report.X_test is None) and (report.y_test is None))
        }
        if len(test_dataset_hashes) > 1:
            raise ValueError("Expected all estimators to have the same testing data.")

        ml_tasks = {report: report._ml_task for report in reports}
        if len(set(ml_tasks.values())) > 1:
            raise ValueError(
                f"Expected all estimators to have the same ML usecase; got {ml_tasks}"
            )

        if report_names is None:
            self.report_names_ = [report.estimator_name_ for report in reports]
        else:
            self.report_names_ = report_names

        self.estimator_reports_ = reports

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
        self._ml_task = self.estimator_reports_[0]._ml_task

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
        for report in self.estimator_reports_:
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

        total_estimators = len(self.estimator_reports_)
        progress.update(main_task, total=total_estimators)

        for estimator_report in self.estimator_reports_:
            # Pass the progress manager to child tasks
            estimator_report._parent_progress = progress
            estimator_report.cache_predictions(
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
            for report in self.estimator_reports_
        ]

    def plot_perf_against_time(
        self,
        perf_metric: str,
        data_source: Literal["test", "train", "X_y"] = "test",
        time_metric: Literal["fit", "predict"] = "predict",
    ):
        """
        Plot a given performance metric against a time metric.

        Parameters
        ----------
        perf_metric : str

        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        perf_metric : str

        time_metric: {"fit", "predict"}, default = "predict"
            The time metric to use in the plot.


        Returns
        -------
        A matplotlib plot.

        """
        # Border cases to handle:
        # - what if a metrics in not computed on all the estimators?
        # - what if a metrics need pos_label?
        # - what if time_metric = "fit", and data_source != "train"?

        # Question
        # should this become an accessor method, e.g. `plots`,
        # the equivalent to `metrics`?

        # TODO
        # - add example
        # - add test
        # - add kwargs

        scatter_data = self.metrics.report_metrics().T.reset_index()
        scatter_data.plot(
            kind="scatter",
            x="Fit time",
            y="Brier score",
            title="Performance vs Time (s)",
        )

        # Add labels to the points with a small offset
        text = scatter_data["Estimator"]
        x = scatter_data["Fit time"]
        y = scatter_data["Brier score"]
        for label, x_coord, y_coord in zip(text, x, y):
            plt.annotate(
                label,
                (x_coord, y_coord),
                textcoords="offset points",
                xytext=(10, 0),
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    edgecolor="gray",
                    facecolor="white",
                    alpha=0.7,
                ),
            )
        plt.tight_layout()

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
