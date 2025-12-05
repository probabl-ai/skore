from __future__ import annotations

import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from rich.panel import Panel
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import check_cv
from sklearn.pipeline import Pipeline

from skore._externals._pandas_accessors import DirNamesMixin
from skore._externals._sklearn_compat import _safe_indexing
from skore._sklearn._base import _BaseReport
from skore._sklearn._estimator.report import EstimatorReport
from skore._sklearn.find_ml_task import _find_ml_task
from skore._sklearn.types import _DEFAULT, MLTask, PositiveLabel, SKLearnCrossValidator
from skore._utils._cache import Cache
from skore._utils._fixes import _validate_joblib_parallel_params
from skore._utils._parallel import Parallel, delayed
from skore._utils._progress_bar import progress_decorator

if TYPE_CHECKING:
    from collections.abc import Iterable

    from skore._sklearn._cross_validation.feature_importance_accessor import (
        _FeatureImportanceAccessor,
    )
    from skore._sklearn._cross_validation.metrics_accessor import _MetricsAccessor


def _generate_estimator_report(
    estimator: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike | None,
    pos_label: PositiveLabel | None,
    train_indices: ArrayLike,
    test_indices: ArrayLike,
) -> EstimatorReport | KeyboardInterrupt | Exception:
    if y is None:
        # In the case of clustering, we do not have y
        y_train = None
        y_test = None
    else:
        y_train = _safe_indexing(y, train_indices)
        y_test = _safe_indexing(y, test_indices)
    try:
        return EstimatorReport(
            estimator,
            fit=True,
            X_train=_safe_indexing(X, train_indices),
            y_train=y_train,
            X_test=_safe_indexing(X, test_indices),
            y_test=y_test,
            pos_label=pos_label,
        )
    except (KeyboardInterrupt, Exception) as e:
        return e


class CrossValidationReport(_BaseReport, DirNamesMixin):
    """Report for cross-validation results.

    Upon initialization, `CrossValidationReport` will clone ``estimator`` according to
    ``splitter`` and fit the generated estimators. The fitting is done in parallel,
    and can be interrupted: the estimators that have been fitted can be accessed even if
    the full cross-validation process did not complete. In particular,
    `KeyboardInterrupt` exceptions are swallowed and will only interrupt the
    cross-validation process, rather than the entire program.

    Refer to the :ref:`cross_validation_report` section of the user guide for more
    details.

    Parameters
    ----------
    estimator : estimator object
        Estimator to make the cross-validation report from.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of supervised learning.

    pos_label : int, float, bool or str, default=None
        For binary classification, the positive class. If `None` and the target labels
        are `{0, 1}` or `{-1, 1}`, the positive class is set to `1`. For other labels,
        some metrics might raise an error if `pos_label` is not defined.

    splitter : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for `splitter` are:

        - int, to specify the number of splits in a `(Stratified)KFold`,
        - a scikit-learn :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer to scikit-learn's :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        When accessing some methods of the `CrossValidationReport`, the `n_jobs`
        parameter is used to parallelize the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    estimator_ : estimator object
        The cloned or copied estimator.

    estimator_name_ : str
        The name of the estimator.

    estimator_reports_ : list of EstimatorReport
        The estimator reports for each split.

    See Also
    --------
    skore.EstimatorReport
        Report for a fitted estimator.

    skore.ComparisonReport
        Report of comparison between estimators.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(random_state=42)
    >>> estimator = LogisticRegression()
    >>> from skore import CrossValidationReport
    >>> report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]] = {
        "metrics": {"name": "metrics"},
        "feature_importance": {"name": "feature_importance"},
    }
    metrics: _MetricsAccessor
    feature_importance: _FeatureImportanceAccessor

    def __init__(
        self,
        estimator: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike | None = None,
        pos_label: PositiveLabel | None = None,
        splitter: int | SKLearnCrossValidator | Generator | None = None,
        n_jobs: int | None = None,
    ) -> None:
        # used to know if a parent launch a progress bar manager
        self._progress_info: dict[str, Any] | None = None

        self._estimator = clone(estimator)

        # private storage to be able to invalidate the cache when the user alters
        # those attributes
        self._X = X
        self._y = y
        self._pos_label = pos_label
        self._splitter = check_cv(splitter, y, classifier=is_classifier(estimator))
        self._split_indices = tuple(self._splitter.split(self._X, self._y))
        self.n_jobs = n_jobs

        self.estimator_reports_: list[EstimatorReport] = self._fit_estimator_reports()

        self._rng = np.random.default_rng(time.time_ns())
        self._hash = self._rng.integers(
            low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max
        )
        self._cache = Cache()
        self._ml_task = _find_ml_task(
            y, estimator=self.estimator_reports_[0]._estimator
        )

    @progress_decorator(
        description=lambda self: (
            f"Processing cross-validation\nfor {self.estimator_name_}"
        )
    )
    def _fit_estimator_reports(self) -> list[EstimatorReport]:
        """Fit the estimator reports.

        This function is created to be able to use the progress bar. It works well
        with the patch of `rich` in VS Code.

        Returns
        -------
        estimator_reports : list of EstimatorReport
            The estimator reports.
        """
        assert self._progress_info is not None, (
            "The rich Progress class was not initialized."
        )
        progress = self._progress_info["current_progress"]
        task = self._progress_info["current_task"]

        progress.update(task, total=len(self.split_indices))

        parallel = Parallel(
            **_validate_joblib_parallel_params(
                n_jobs=self.n_jobs, return_as="generator"
            )
        )
        # do not split the data to take advantage of the memory mapping
        generator = parallel(
            delayed(_generate_estimator_report)(
                clone(self._estimator),
                self._X,
                self._y,
                self._pos_label,
                train_indices,
                test_indices,
            )
            for (train_indices, test_indices) in self.split_indices
        )

        estimator_reports = []
        for report in generator:
            estimator_reports.append(report)
            progress.update(task, advance=1, refresh=True)

        warn_msg = None
        if not any(isinstance(report, EstimatorReport) for report in estimator_reports):
            traceback_msg = "\n".join(str(exc) for exc in estimator_reports)
            raise RuntimeError(
                "Cross-validation failed: no estimators were successfully fitted. "
                "Please check your data, estimator, or cross-validation setup.\n"
                f"Traceback: \n{traceback_msg}"
            )
        elif any(isinstance(report, Exception) for report in estimator_reports):
            msg_traceback = "\n".join(
                str(exc) for exc in estimator_reports if isinstance(exc, Exception)
            )
            warn_msg = (
                "Cross-validation process was interrupted by an error before "
                "all estimators could be fitted; CrossValidationReport object "
                "might not contain all the expected results.\n"
                f"Traceback: \n{msg_traceback}"
            )
            estimator_reports = [
                report
                for report in estimator_reports
                if not isinstance(report, Exception)
            ]
        if any(isinstance(report, KeyboardInterrupt) for report in estimator_reports):
            warn_msg = (
                "Cross-validation process was interrupted manually before all "
                "estimators could be fitted; CrossValidationReport object "
                "might not contain all the expected results."
            )
            estimator_reports = [
                report
                for report in estimator_reports
                if not isinstance(report, KeyboardInterrupt)
            ]

        if warn_msg is not None:
            from skore import console  # avoid circular import

            console.print(
                Panel(
                    title="Cross-validation interrupted",
                    renderable=warn_msg,
                    style="orange1",
                    border_style="cyan",
                )
            )

        return estimator_reports

    def clear_cache(self) -> None:
        """Clear the cache.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, splitter=2)
        >>> report.cache_predictions()
        >>> report.clear_cache()
        >>> report._cache
        {}
        """
        for report in self.estimator_reports_:
            report.clear_cache()

        self._cache = Cache()

    @progress_decorator(description="Cross-validation predictions")
    def cache_predictions(
        self,
        response_methods: str = "auto",
        n_jobs: int | None = None,
    ) -> None:
        """Cache the predictions for sub-estimators reports.

        Parameters
        ----------
        response_methods : {"auto", "predict", "predict_proba", "decision_function"},\
                default="auto
            The methods to use to compute the predictions.

        n_jobs : int, default=None
            The number of jobs to run in parallel. If `None`, we use the `n_jobs`
            parameter when initializing `CrossValidationReport`.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, splitter=2)
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

        for split_idx, estimator_report in enumerate(self.estimator_reports_, 1):
            # Share the parent's progress bar with child report
            estimator_report._progress_info = {
                "current_progress": progress,
                "split_info": {"current": split_idx, "total": total_estimators},
            }

            # Update the progress bar description to include the split number
            progress.update(
                main_task,
                description=(
                    "Cross-validation predictions for split "
                    f"#{split_idx}/{total_estimators}"
                ),
            )

            # Call cache_predictions without printing a separate message
            estimator_report.cache_predictions(
                response_methods=response_methods, n_jobs=n_jobs
            )
            progress.update(main_task, advance=1, refresh=True)

    def get_predictions(
        self,
        *,
        data_source: Literal["train", "test", "X_y"],
        response_method: Literal[
            "predict", "predict_proba", "decision_function"
        ] = "predict",
        X: ArrayLike | None = None,
        pos_label: PositiveLabel | None = _DEFAULT,
    ) -> list[ArrayLike]:
        """Get estimator's predictions.

        This method has the advantage to reload from the cache if the predictions
        were already computed in a previous call.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the train set provided when creating the report and the target
              variable.

        response_method : {"predict", "predict_proba", "decision_function"}, \
                default="predict"
            The response method to use to get the predictions.

        X : array-like of shape (n_samples, n_features), optional
            When `data_source` is "X_y", the input features on which to compute the
            response method.

        pos_label : int, float, bool, str or None, default=_DEFAULT
            The label to consider as the positive class when computing predictions in
            binary classification cases. By default, the positive class is set to the
            one provided when creating the report. If `None`, `estimator_.classes_[1]`
            is used as positive label.

            When `pos_label` is equal to `estimator_.classes_[0]`, it will be equivalent
            to `estimator_.predict_proba(X)[:, 0]` for `response_method="predict_proba"`
            and `-estimator_.decision_function(X)` for
            `response_method="decision_function"`.

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
        >>> from sklearn.linear_model import LogisticRegression
        >>> X, y = make_classification(random_state=42)
        >>> estimator = LogisticRegression()
        >>> from skore import CrossValidationReport
        >>> report = CrossValidationReport(estimator, X=X, y=y, splitter=2)
        >>> predictions = report.get_predictions(data_source="test")
        >>> print([split_predictions.shape for split_predictions in predictions])
        [(50,), (50,)]
        """
        if data_source not in ("train", "test", "X_y"):
            raise ValueError(
                f"Invalid data source: {data_source}. Valid data sources are "
                "'train', 'test' and 'X_y'."
            )
        return [
            report.get_predictions(
                data_source=data_source,
                response_method=response_method,
                X=X,
                pos_label=pos_label,
            )
            for report in self.estimator_reports_
        ]

    @property
    def ml_task(self) -> MLTask:
        return self._ml_task

    @property
    def estimator(self) -> BaseEstimator:
        return self._estimator

    @property
    def estimator_(self) -> BaseEstimator:
        return self._estimator

    @property
    def estimator_name_(self) -> str:
        if isinstance(self._estimator, Pipeline):
            name = self._estimator[-1].__class__.__name__
        else:
            name = self._estimator.__class__.__name__
        return name

    @property
    def X(self) -> ArrayLike:
        return self._X

    @property
    def y(self) -> ArrayLike | None:
        return self._y

    @property
    def splitter(self) -> SKLearnCrossValidator:
        return self._splitter

    @property
    def split_indices(self) -> tuple[tuple[Iterable[int], Iterable[int]]]:
        return self._split_indices

    @property
    def pos_label(self) -> PositiveLabel | None:
        return self._pos_label

    @pos_label.setter
    def pos_label(self, value: PositiveLabel | None) -> None:
        raise AttributeError(
            "The pos_label attribute is immutable. "
            f"Call the constructor of {self.__class__.__name__} to create a new report."
        )

    ####################################################################################
    # Methods related to the help and repr
    ####################################################################################

    def _get_help_panel_title(self) -> str:
        return (
            f"[bold cyan]Tools to diagnose estimator {self.estimator_name_}[/bold cyan]"
        )

    def _get_help_legend(self) -> str:
        return (
            "[cyan](↗︎)[/cyan] higher is better [orange1](↘︎)[/orange1] lower is better"
        )

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{self.__class__.__name__}(estimator={self.estimator_}, ...)"
