import copy
import time
import warnings
from itertools import product
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from skore.externals._pandas_accessors import DirNamesMixin
from skore.externals._sklearn_compat import is_clusterer
from skore.sklearn._base import _BaseReport, _get_cached_response_values
from skore.sklearn.find_ml_task import _find_ml_task
from skore.utils._parallel import Parallel, delayed
from skore.utils._progress_bar import progress_decorator

if TYPE_CHECKING:
    from skore.sklearn._estimator.metrics_accessor import _MetricsAccessor


class EstimatorReport(_BaseReport, DirNamesMixin):
    """Report for a fitted estimator.

    This class provides a set of tools to quickly validate and inspect a scikit-learn
    compatible estimator.

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

    Attributes
    ----------
    estimator_ : estimator object
        The cloned or copied estimator.

    estimator_name_ : str
        The name of the estimator.

    See Also
    --------
    skore.sklearn.cross_validation.report.CrossValidationReport
        Report of cross-validation results.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> estimator = LogisticRegression().fit(X_train, y_train)
    >>> from skore import EstimatorReport
    >>> report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]] = {
        "metrics": {"name": "metrics"},
        "feature_importance": {"name": "feature_importance"},
    }
    metrics: "_MetricsAccessor"

    @staticmethod
    def _fit_estimator(
        estimator: BaseEstimator,
        X_train: Union[ArrayLike, None],
        y_train: Union[ArrayLike, None],
    ) -> BaseEstimator:
        if X_train is None or (y_train is None and not is_clusterer(estimator)):
            raise ValueError(
                "The training data is required to fit the estimator. "
                "Please provide both X_train and y_train."
            )
        return clone(estimator).fit(X_train, y_train)

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
        fit: Union[Literal["auto"], bool] = "auto",
        X_train: Optional[ArrayLike] = None,
        y_train: Optional[ArrayLike] = None,
        X_test: Optional[ArrayLike] = None,
        y_test: Optional[ArrayLike] = None,
    ) -> None:
        # used to know if a parent launch a progress bar manager
        self._progress_info: Optional[dict[str, Any]] = None
        self._parent_progress = None

        if fit == "auto":
            try:
                check_is_fitted(estimator)
                self._estimator = self._copy_estimator(estimator)
            except NotFittedError:
                self._estimator = self._fit_estimator(estimator, X_train, y_train)
        elif fit is True:
            self._estimator = self._fit_estimator(estimator, X_train, y_train)
        else:  # fit is False
            self._estimator = self._copy_estimator(estimator)

        # private storage to be able to invalidate the cache when the user alters
        # those attributes
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test

        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize/reset the random number generator, hash, and cache."""
        self._rng = np.random.default_rng(time.time_ns())
        self._hash = self._rng.integers(
            low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max
        )
        self._cache: dict[tuple[Any, ...], Any] = {}
        self._ml_task = _find_ml_task(self._y_test, estimator=self._estimator)

    # NOTE:
    # For the moment, we do not allow to alter the estimator and the training data.
    # For the validation set, we allow it and we invalidate the cache.

    def clear_cache(self) -> None:
        """Clear the cache.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_breast_cancer(return_X_y=True), random_state=0
        ... )
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(
        ...     classifier,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> report.cache_predictions()
        >>> report.clear_cache()
        >>> report._cache
        {}
        """
        self._cache = {}

    @progress_decorator(description="Caching predictions")
    def cache_predictions(
        self,
        response_methods: Union[Literal["auto"], list[str]] = "auto",
        n_jobs: Optional[int] = None,
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
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_breast_cancer(return_X_y=True), random_state=0
        ... )
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(
        ...     classifier,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
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

        parallel = Parallel(n_jobs=n_jobs, return_as="generator", require="sharedmem")
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
        for _ in generator:
            progress.update(task, advance=1, refresh=True)

    @property
    def estimator_(self) -> BaseEstimator:
        return self._estimator

    @estimator_.setter
    def estimator_(self, value):
        raise AttributeError(
            "The estimator attribute is immutable. "
            f"Call the constructor of {self.__class__.__name__} to create a new report."
        )

    @property
    def X_train(self) -> Optional[ArrayLike]:
        return self._X_train

    @X_train.setter
    def X_train(self, value):
        raise AttributeError(
            "The X_train attribute is immutable. "
            f"Call the constructor of {self.__class__.__name__} to create a new report."
        )

    @property
    def y_train(self) -> Optional[ArrayLike]:
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        raise AttributeError(
            "The y_train attribute is immutable. "
            f"Call the constructor of {self.__class__.__name__} to create a new report."
        )

    @property
    def X_test(self) -> Optional[ArrayLike]:
        return self._X_test

    @X_test.setter
    def X_test(self, value):
        self._X_test = value
        self._initialize_state()

    @property
    def y_test(self) -> Optional[ArrayLike]:
        return self._y_test

    @y_test.setter
    def y_test(self, value):
        self._y_test = value
        self._initialize_state()

    @property
    def estimator_name_(self) -> str:
        if isinstance(self._estimator, Pipeline):
            name = self._estimator[-1].__class__.__name__
        else:
            name = self._estimator.__class__.__name__
        return name

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
