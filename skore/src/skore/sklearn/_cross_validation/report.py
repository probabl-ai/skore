import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from rich.panel import Panel
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import check_cv
from sklearn.pipeline import Pipeline

from skore.externals._pandas_accessors import DirNamesMixin
from skore.externals._sklearn_compat import _safe_indexing
from skore.sklearn._base import _BaseReport
from skore.sklearn._estimator.report import EstimatorReport
from skore.sklearn.find_ml_task import _find_ml_task
from skore.sklearn.types import SKLearnCrossValidator
from skore.utils._parallel import Parallel, delayed
from skore.utils._progress_bar import progress_decorator

if TYPE_CHECKING:
    from skore.sklearn._cross_validation.metrics_accessor import _MetricsAccessor


def _generate_estimator_report(
    estimator: BaseEstimator,
    X: ArrayLike,
    y: Optional[ArrayLike],
    train_indices: ArrayLike,
    test_indices: ArrayLike,
) -> EstimatorReport:
    return EstimatorReport(
        estimator,
        fit=True,
        X_train=_safe_indexing(X, train_indices),
        y_train=_safe_indexing(y, train_indices),
        X_test=_safe_indexing(X, test_indices),
        y_test=_safe_indexing(y, test_indices),
    )


class CrossValidationReport(_BaseReport, DirNamesMixin):
    """Report for cross-validation results.

    Upon initialization, `CrossValidationReport` will clone ``estimator`` according to
    ``cv_splitter`` and fit the generated estimators. The fitting is done in parallel,
    and can be interrupted: the estimators that have been fitted can be accessed even if
    the full cross-validation process did not complete. In particular,
    `KeyboardInterrupt` exceptions are swallowed and will only interrupt the
    cross-validation process, rather than the entire program.

    Parameters
    ----------
    estimator : estimator object
        Estimator to make the cross-validation report from.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of supervised learning.

    cv_splitter : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for `cv_splitter` are:

        - int, to specify the number of folds in a `(Stratified)KFold`,
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

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(random_state=42)
    >>> estimator = LogisticRegression()
    >>> from skore import CrossValidationReport
    >>> report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=2)
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]] = {
        "metrics": {"name": "metrics"},
    }
    metrics: "_MetricsAccessor"

    def __init__(
        self,
        estimator: BaseEstimator,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        cv_splitter: Optional[Union[int, SKLearnCrossValidator, Generator]] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        # used to know if a parent launch a progress bar manager
        self._progress_info: Optional[dict[str, Any]] = None
        self._parent_progress = None

        self._estimator = clone(estimator)

        # private storage to be able to invalidate the cache when the user alters
        # those attributes
        self._X = X
        self._y = y
        self._cv_splitter = check_cv(
            cv_splitter, y, classifier=is_classifier(estimator)
        )
        self.n_jobs = n_jobs

        self.estimator_reports_ = self._fit_estimator_reports()

        self._rng = np.random.default_rng(time.time_ns())
        self._hash = self._rng.integers(
            low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max
        )
        self._cache: dict[tuple[Any, ...], Any] = {}
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

        n_splits = self._cv_splitter.get_n_splits(self._X, self._y)
        progress.update(task, total=n_splits)

        parallel = Parallel(n_jobs=self.n_jobs, return_as="generator")
        # do not split the data to take advantage of the memory mapping
        generator = parallel(
            delayed(_generate_estimator_report)(
                clone(self._estimator),
                self._X,
                self._y,
                train_indices,
                test_indices,
            )
            for train_indices, test_indices in self._cv_splitter.split(self._X, self._y)
        )

        estimator_reports = []
        try:
            for report in generator:
                estimator_reports.append(report)
                progress.update(task, advance=1, refresh=True)
        except (Exception, KeyboardInterrupt) as e:
            from skore import console  # avoid circular import

            if isinstance(e, KeyboardInterrupt):
                message = (
                    "Cross-validation process was interrupted manually before all "
                    "estimators could be fitted; CrossValidationReport object "
                    "might not contain all the expected results."
                )
            else:
                message = (
                    "Cross-validation process was interrupted by an error before "
                    "all estimators could be fitted; CrossValidationReport object "
                    "might not contain all the expected results. "
                    f"Traceback: \n{e}"
                )

            console.print(
                Panel(
                    title="Cross-validation interrupted",
                    renderable=message,
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
        >>> report = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        >>> report.cache_predictions()
        >>> report.clear_cache()
        >>> report._cache
        {}
        """
        for report in self.estimator_reports_:
            report.clear_cache()
        self._cache = {}

    @progress_decorator(description="Cross-validation predictions")
    def cache_predictions(
        self,
        response_methods: str = "auto",
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
            parameter when initializing `CrossValidationReport`.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
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

    @property
    def estimator_(self) -> BaseEstimator:
        return self._estimator

    @estimator_.setter
    def estimator_(self, value: Any) -> None:
        raise AttributeError(
            "The estimator attribute is immutable. "
            f"Call the constructor of {self.__class__.__name__} to create a new report."
        )

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

    @X.setter
    def X(self, value: Any) -> None:
        raise AttributeError(
            "The X attribute is immutable. "
            f"Call the constructor of {self.__class__.__name__} to create a new report."
        )

    @property
    def y(self) -> Optional[ArrayLike]:
        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        raise AttributeError(
            "The y attribute is immutable. "
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
