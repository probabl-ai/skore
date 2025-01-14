import time

import joblib
import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.model_selection import check_cv
from sklearn.pipeline import Pipeline

from skore.externals._pandas_accessors import DirNamesMixin
from skore.externals._sklearn_compat import _safe_indexing
from skore.sklearn._base import _BaseReport
from skore.sklearn._estimator.report import EstimatorReport
from skore.sklearn.find_ml_task import _find_ml_task
from skore.utils._progress_bar import progress_decorator


def _generate_estimator_report(estimator, X, y, train_indices, test_indices):
    return EstimatorReport(
        estimator,
        fit=True,
        X_train=_safe_indexing(X, train_indices),
        y_train=_safe_indexing(y, train_indices),
        X_test=_safe_indexing(X, test_indices),
        y_test=_safe_indexing(y, test_indices),
    )


class CrossValidationReport(_BaseReport, DirNamesMixin):
    """Reporter for cross-validation results.

    Parameters
    ----------
    estimator : estimator object
        Estimator to make report from.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        The target variable to try to predict in the case of supervised learning.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    """

    _ACCESSOR_CONFIG = {
        "metrics": {"icon": ":straight_ruler:", "name": "metrics"},
    }

    def __init__(
        self,
        estimator,
        X,
        y=None,
        cv=None,
        n_jobs=None,
    ):
        self._parent_progress = None  # used for the different progress bars
        self._estimator = clone(estimator)

        # private storage to be able to invalidate the cache when the user alters
        # those attributes
        self._X = X
        self._y = y
        self._cv = check_cv(cv, y, classifier=is_classifier(estimator))
        self._n_jobs = n_jobs

        self.estimator_reports = self._fit_estimator_reports()

        self._rng = np.random.default_rng(time.time_ns())
        self._hash = self._rng.integers(
            low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max
        )
        self._cache = {}
        self._ml_task = _find_ml_task(y, estimator=self.estimator_reports[0].estimator)

    @progress_decorator(
        description=lambda self: (
            f"Processing cross-validation\nfor {self.estimator_name}"
        )
    )
    def _fit_estimator_reports(self):
        """Fit the estimator reports.

        This function is created to be able to use the progress bar. It works well
        with the patch of `rich` in VS Code.

        Returns
        -------
        estimator_reports : list of EstimatorReport
            The estimator reports.
        """
        progress = self._progress_info["current_progress"]
        task = self._progress_info["current_task"]

        n_splits = self._cv.get_n_splits(self._X, self._y)
        progress.update(task, total=n_splits)

        parallel = joblib.Parallel(n_jobs=self._n_jobs, return_as="generator_unordered")
        # do not split the data to take advantage of the memory mapping
        generator = parallel(
            joblib.delayed(_generate_estimator_report)(
                clone(self._estimator),
                self._X,
                self._y,
                train_indices,
                test_indices,
            )
            for train_indices, test_indices in self._cv.split(self._X, self._y)
        )

        estimator_reports = []
        for report in generator:
            estimator_reports.append(report)
            progress.update(task, advance=1, refresh=True)

        return estimator_reports

    def clean_cache(self):
        """Clean the cache."""
        for report in self.estimator_reports:
            report.clean_cache()
        self._cache = {}

    @progress_decorator(description="Cross-validation predictions")
    def cache_predictions(self, response_methods="auto"):
        """Cache the predictions for sub-estimators reports.

        Parameters
        ----------
        response_methods : {"auto", "predict", "predict_proba", "decision_function"},\
                default="auto
            The methods to use to compute the predictions.
        """
        progress = self._progress_info["current_progress"]
        main_task = self._progress_info["current_task"]

        total_estimators = len(self.estimator_reports)
        progress.update(main_task, total=total_estimators)

        for estimator_report in self.estimator_reports:
            # Pass the progress manager to child tasks
            estimator_report._parent_progress = progress
            estimator_report.cache_predictions(
                response_methods=response_methods, n_jobs=self._n_jobs
            )
            progress.update(main_task, advance=1, refresh=True)

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        raise AttributeError(
            "The estimator attribute is immutable. "
            f"Call the constructor of {self.__class__.__name__} to create a new report."
        )

    @property
    def estimator_name(self):
        if isinstance(self.estimator, Pipeline):
            name = self.estimator[-1].__class__.__name__
        else:
            name = self.estimator.__class__.__name__
        return name

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        raise AttributeError(
            "The X attribute is immutable. "
            "Please use the `from_unfitted_estimator` method to create a new report."
        )

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        raise AttributeError(
            "The y attribute is immutable. "
            "Please use the `from_unfitted_estimator` method to create a new report."
        )

    ####################################################################################
    # Methods related to the help and repr
    ####################################################################################

    def _get_help_panel_title(self):
        return (
            f"[bold cyan]📓 Tools to diagnose estimator "
            f"{self.estimator_name}[/bold cyan]"
        )

    def _get_help_legend(self):
        return (
            "[cyan](↗︎)[/cyan] higher is better [orange1](↘︎)[/orange1] lower is better"
        )
