from __future__ import annotations

import copy
import time
import warnings
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from skore._externals._pandas_accessors import DirNamesMixin
from skore._externals._sklearn_compat import is_clusterer
from skore._sklearn._base import _BaseReport, _get_cached_response_values
from skore._sklearn.find_ml_task import _find_ml_task
from skore._sklearn.types import _DEFAULT, MLTask, PositiveLabel
from skore._utils._cache import Cache
from skore._utils._fixes import _validate_joblib_parallel_params
from skore._utils._measure_time import MeasureTime
from skore._utils._parallel import Parallel, delayed
from skore._utils._progress_bar import progress_decorator

if TYPE_CHECKING:
    from skore._sklearn._estimator.data_accessor import _DataAccessor
    from skore._sklearn._estimator.feature_importance_accessor import (
        _FeatureImportanceAccessor,
    )
    from skore._sklearn._estimator.metrics_accessor import _MetricsAccessor


class EstimatorReport(_BaseReport, DirNamesMixin):
    """Report for a fitted estimator.

    This class provides a set of tools to quickly validate and inspect a scikit-learn
    compatible estimator.

    Refer to the :ref:`estimator_report` section of the user guide for more details.

    Parameters
    ----------
    estimator : estimator object
        Estimator to make the report from. When the estimator is not fitted,
        it is deep-copied to avoid side-effects. If it is fitted, it is cloned instead.

    fit : {"auto", True, False}, default="auto"
        Whether to fit the estimator on the training data. If "auto", the estimator
        is fitted only if the training data is provided.

    X_train : {array-like, sparse matrix} of shape (n_samples, n_features) or \
            None
        Training data.

    y_train : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        Training target.

    X_test : {array-like, sparse matrix} of shape (n_samples, n_features) or None
        Testing data. It should have the same structure as the training data.

    y_test : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        Testing target.

    pos_label : int, float, bool or str, default=None
        For binary classification, the positive class. If `None` and the target labels
        are `{0, 1}` or `{-1, 1}`, the positive class is set to `1`. For other labels,
        some metrics might raise an error if `pos_label` is not defined.

    Attributes
    ----------
    estimator_ : estimator object
        The cloned or copied estimator.

    estimator_name_ : str
        The name of the estimator.

    fit_time_ : float or None
        The time taken to fit the estimator, in seconds. If the estimator is not
        internally fitted, the value is `None`.

    See Also
    --------
    skore.CrossValidationReport
        Report of cross-validation results.

    skore.ComparisonReport
        Report of comparison between estimators.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from skore import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(random_state=42)
    >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
    >>> estimator = LogisticRegression()
    >>> from skore import EstimatorReport
    >>> report = EstimatorReport(estimator, **split_data)
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]] = {
        "metrics": {"name": "metrics"},
        "feature_importance": {"name": "feature_importance"},
        "data": {"name": "data"},
    }

    metrics: _MetricsAccessor
    feature_importance: _FeatureImportanceAccessor
    data: _DataAccessor

    @staticmethod
    def _fit_estimator(
        estimator: BaseEstimator,
        X_train: ArrayLike | None,
        y_train: ArrayLike | None,
    ) -> tuple[BaseEstimator, float]:
        if X_train is None or (y_train is None and not is_clusterer(estimator)):
            raise ValueError(
                "The training data is required to fit the estimator. "
                "Please provide both X_train and y_train."
            )
        estimator_ = clone(estimator)
        with MeasureTime() as fit_time:
            estimator_.fit(X_train, y_train)
        return estimator_, fit_time()

    @classmethod
    def _copy_estimator(cls, estimator: BaseEstimator) -> BaseEstimator:
        try:
            return copy.deepcopy(estimator)
        except Exception as e:
            warnings.warn(
                "Deepcopy failed; using estimator as-is. "
                "Be aware that modifying the estimator outside of "
                f"{cls.__name__} will modify the internal estimator. "
                "Consider using a FrozenEstimator from scikit-learn to prevent this. "
                f"Original error: {e}",
                stacklevel=1,
            )
        return estimator

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        fit: Literal["auto"] | bool = "auto",
        X_train: ArrayLike | None = None,
        y_train: ArrayLike | None = None,
        X_test: ArrayLike | None = None,
        y_test: ArrayLike | None = None,
        pos_label: PositiveLabel | None = None,
    ) -> None:
        # used to know if a parent launch a progress bar manager
        self._progress_info: dict[str, Any] | None = None
        self._fit = fit

        fit_time: float | None = None
        if fit == "auto":
            try:
                check_is_fitted(estimator)
                self._estimator = self._copy_estimator(estimator)
            except NotFittedError:
                self._estimator, fit_time = self._fit_estimator(
                    estimator, X_train, y_train
                )
        elif fit is True:
            self._estimator, fit_time = self._fit_estimator(estimator, X_train, y_train)
        else:  # fit is False
            self._estimator = self._copy_estimator(estimator)

        # private storage to be able to invalidate the cache when the user alters
        # those attributes
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._pos_label = pos_label
        self.fit_time_ = fit_time

        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize/reset the random number generator, hash, and cache."""
        self._rng = np.random.default_rng(time.time_ns())
        self._hash = self._rng.integers(
            low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max
        )
        self._cache = Cache()
        self._ml_task = _find_ml_task(self._y_test, estimator=self._estimator)

    # NOTE:
    # For the moment, we do not allow to alter the estimator and the training data.
    # For the validation set, we allow it and we invalidate the cache.

    def clear_cache(self) -> None:
        """Clear the cache.

        Note that the cache might not be empty after this method is run as some
        values need to be kept, such as the fit time.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data)
        >>> report.cache_predictions()
        >>> report.clear_cache()
        >>> report._cache
        {}
        """
        self._cache = Cache()

    @progress_decorator(description="Caching predictions")
    def cache_predictions(
        self,
        response_methods: Literal["auto"] | list[str] = "auto",
        n_jobs: int | None = None,
    ) -> None:
        """Cache estimator's predictions.

        Parameters
        ----------
        response_methods : "auto" or list of str, default="auto"
            The response methods to precompute. If "auto", the response methods are
            inferred from the ml task: for classification we compute the response of
            the `predict_proba`, `decision_function` and `predict` methods; for
            regression we compute the response of the `predict` method.

        n_jobs : int or None, default=None
            The number of jobs to run in parallel. None means 1 unless in a
            joblib.parallel_backend context. -1 means using all processors.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data)
        >>> report.cache_predictions()
        >>> report._cache
        {...}
        """
        if self._ml_task in ("binary-classification", "multiclass-classification"):
            if response_methods == "auto":
                response_methods = ["predict"]
                if hasattr(self._estimator, "predict_proba"):
                    response_methods += ["predict_proba"]
                if hasattr(self._estimator, "decision_function"):
                    response_methods += ["decision_function"]
            pos_labels = self._estimator.classes_.tolist() + [None]
        else:
            if response_methods == "auto":
                response_methods = ["predict"]
            pos_labels = [None]

        data_sources = [("test", self._X_test)]
        if self._X_train is not None:
            data_sources += [("train", self._X_train)]

        parallel = Parallel(
            **_validate_joblib_parallel_params(n_jobs=n_jobs, return_as="generator")
        )
        generator = parallel(
            delayed(_get_cached_response_values)(
                cache=self._cache,
                estimator_hash=self._hash,
                estimator=self._estimator,
                X=X,
                response_method=response_method,
                pos_label=pos_label,
                data_source=data_source,
            )
            for response_method, pos_label, (data_source, X) in product(
                response_methods, pos_labels, data_sources
            )
        )
        # trigger the computation
        assert self._progress_info is not None, (
            "The rich Progress class was not initialized."
        )
        progress = self._progress_info["current_progress"]
        task = self._progress_info["current_task"]
        total_iterations = len(response_methods) * len(pos_labels) * len(data_sources)
        progress.update(task, total=total_iterations)

        # do not mutate directly `self._cache` during the execution of Parallel
        results_to_cache: dict[tuple[Any, ...], Any] = {}
        for results in generator:
            results_to_cache.update(
                (key, value) for key, value, is_cached in results if not is_cached
            )
            progress.update(task, advance=1, refresh=True)

        if results_to_cache:
            self._cache.update(results_to_cache)

    def get_predictions(
        self,
        *,
        data_source: Literal["train", "test", "X_y"],
        response_method: Literal[
            "predict", "predict_proba", "decision_function"
        ] = "predict",
        X: ArrayLike | None = None,
        pos_label: PositiveLabel | None = _DEFAULT,
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
            - "X_y" : use the provided `X` and `y` to compute the predictions.

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
        np.ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predictions.

        Raises
        ------
        ValueError
            If the data source is invalid.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from skore import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> X, y = make_classification(random_state=42)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator = LogisticRegression()
        >>> from skore import EstimatorReport
        >>> report = EstimatorReport(estimator, **split_data)
        >>> predictions = report.get_predictions(data_source="test")
        >>> predictions.shape
        (25,)
        """
        if pos_label is _DEFAULT:
            pos_label = self.pos_label

        if data_source == "test":
            X_ = self._X_test
        elif data_source == "train":
            X_ = self._X_train
        elif data_source == "X_y":
            if X is None:
                raise ValueError(
                    "The `X` parameter is required when `data_source` is 'X_y'."
                )
            X_ = X
        else:
            raise ValueError(f"Invalid data source: {data_source}")

        results = _get_cached_response_values(
            cache=self._cache,
            estimator_hash=int(self._hash),
            estimator=self._estimator,
            X=X_,
            response_method=response_method,
            pos_label=pos_label,
            data_source=data_source,
        )
        for key, value, is_cached in results:
            if not is_cached:
                self._cache[key] = value
        return results[0][1]  # return the predictions only

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
    def X_train(self) -> ArrayLike | None:
        return self._X_train

    @property
    def y_train(self) -> ArrayLike | None:
        return self._y_train

    @property
    def X_test(self) -> ArrayLike | None:
        return self._X_test

    @X_test.setter
    def X_test(self, value):
        self._X_test = value
        self._initialize_state()

    @property
    def y_test(self) -> ArrayLike | None:
        return self._y_test

    @y_test.setter
    def y_test(self, value):
        self._y_test = value
        self._initialize_state()

    @property
    def pos_label(self) -> PositiveLabel | None:
        return self._pos_label

    @pos_label.setter
    def pos_label(self, value: PositiveLabel | None) -> None:
        self._pos_label = value
        self._initialize_state()

    @property
    def estimator_name_(self) -> str:
        if isinstance(self._estimator, Pipeline):
            name = self._estimator[-1].__class__.__name__
        else:
            name = self._estimator.__class__.__name__
        return name

    @property
    def fit(self) -> Literal["auto"] | bool:
        return self._fit

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
