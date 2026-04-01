from __future__ import annotations

import copy
import html
import uuid
import warnings
from itertools import product
from typing import TYPE_CHECKING, Any, Literal, cast

import joblib
import numpy as np
import skrub
from joblib import Parallel
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils._response import (
    _check_response_method,
    _process_decision_function,
    _process_predict_proba,
)
from sklearn.utils.validation import check_is_fitted

from skore._externals._pandas_accessors import DirNamesMixin
from skore._externals._sklearn_compat import is_clusterer
from skore._sklearn._base import _BaseReport
from skore._sklearn.find_ml_task import _find_ml_task
from skore._sklearn.types import DataSource, PositiveLabel
from skore._utils._cache import Cache
from skore._utils._fixes import _validate_joblib_parallel_params
from skore._utils._measure_time import MeasureTime
from skore._utils._parallel import delayed
from skore._utils._progress_bar import track
from skore._utils.repr.data import get_documentation_url
from skore._utils.repr.html_repr import render_template

if TYPE_CHECKING:
    from skore._sklearn._estimator.data_accessor import _DataAccessor
    from skore._sklearn._estimator.inspection_accessor import (
        _InspectionAccessor,
    )
    from skore._sklearn._estimator.metrics_accessor import _MetricsAccessor


def _compute_predictions(estimator, X, response_method):
    prediction_method = _check_response_method(estimator, response_method)
    with MeasureTime() as predict_time:
        predictions = prediction_method(X)

    return predictions, predict_time()


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

    X_test : {array-like, sparse matrix} of shape (n_samples, n_features)
        Testing data. It should have the same structure as the training data.

    y_test : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Testing target.

    pos_label : int, float, bool or str, default=None
        For binary classification, the positive class to use for metrics and displays
        that need one. If `None`, skore does not infer a default positive class.
        Binary metrics and displays that support it will expose all classes instead.
        This parameter is rejected for non-binary tasks.

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
        "data": {"name": "data"},
        "metrics": {"name": "metrics"},
        "inspection": {"name": "inspection"},
    }

    _report_type: Literal["estimator"] = "estimator"

    metrics: _MetricsAccessor
    inspection: _InspectionAccessor
    data: _DataAccessor

    @staticmethod
    def _fit_estimator(
        estimator: BaseEstimator,
        X_train: ArrayLike | None,
        y_train: ArrayLike | None,
    ) -> tuple[BaseEstimator, float]:
        if X_train is None or y_train is None:
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
        X_test: ArrayLike,
        y_test: ArrayLike,
        pos_label: PositiveLabel | None = None,
    ) -> None:
        self._fit = fit

        if is_clusterer(estimator):
            raise ValueError(
                "Clustering models are not supported yet. Please use a"
                " classification or regression model instead."
            )

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

        # private storage to ensure properties are read-only
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._pos_label = pos_label
        self.fit_time_ = fit_time
        # TODO? Rename class "Cache" to "ThreadSafeDict"?
        self._cache = Cache()
        self._predictions = Cache()
        # NOTE: Reports are immutable so we don't need cache invalidation

        self._ml_task = _find_ml_task(self._y_test, estimator=self._estimator)
        if pos_label is None:
            return
        if self._ml_task != "binary-classification":
            raise ValueError(
                "pos_label is only accepted/used for binary classification"
            )
        labels = self.estimator_.classes_.tolist()
        if pos_label not in labels:
            raise ValueError(
                f"pos_label={pos_label!r} is not a valid label. "
                f"It should be one of: {labels!r}."
            )

    def clear_cache(self) -> None:
        """Clear the cache.

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
        self._predictions = Cache()

    def cache_predictions(
        self,
        response_methods: Literal["auto"] | str | list[str] = "auto",
        data_source: DataSource | Literal["both"] = "both",
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

        data_source : {"test", "train", "both"}, default="both"
            The data source(s) for which to precompute predictions.

            - "test" : cache predictions for the test set only.
            - "train" : cache predictions for the train set only.
            - "both" : cache predictions for both train and test sets when available.

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
        else:
            if response_methods == "auto":
                response_methods = ["predict"]

        data_sources = []
        if data_source in ("test", "both"):
            data_sources += [("test", self._X_test)]
        if data_source in ("train", "both") and self._X_train is not None:
            data_sources += [("train", self._X_train)]

        keys_and_to_compute = [
            (
                (cast(DataSource, ds), response_method),
                (self._estimator, X, response_method),
            )
            for response_method, (ds, X) in product(response_methods, data_sources)
        ]

        keys_and_to_compute = [
            (key, data)
            for key, data in keys_and_to_compute
            if key not in self._predictions
        ]

        if not keys_and_to_compute:
            return

        parallel = Parallel(
            **_validate_joblib_parallel_params(n_jobs=n_jobs, return_as="generator")
        )

        keys, to_compute = zip(*keys_and_to_compute, strict=True)
        # trigger the computation
        for (ds, response_method), (preds, time) in zip(
            keys,
            track(
                parallel(delayed(_compute_predictions)(*args) for args in to_compute),
                description="Caching predictions",
                total=(len(response_methods) * len(data_sources)),
            ),
            strict=True,
        ):
            self._predictions[(ds, response_method)] = preds
            if response_method == "predict":
                self._predictions[(ds, "predict_time")] = time

    def get_predictions(
        self,
        *,
        data_source: Literal["train", "test"],
        response_method: Literal[
            "predict", "predict_proba", "decision_function"
        ] = "predict",
    ) -> ArrayLike:
        """Get estimator's predictions.

        This method has the advantage to reload from the cache if the predictions
        were already computed in a previous call.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        response_method : {"predict", "predict_proba", "decision_function"}, \
                default="predict"
            The response method to use to get the predictions.

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
        pos_label = self.pos_label
        if (
            pos_label is None
            and self.ml_task == "binary-classification"
            and response_method == "decision_function"
        ):
            # we do this to follow scikit-learn convention:
            pos_label = self.estimator_.classes_[-1]
        return self._get_predictions(
            data_source=data_source,
            response_method=response_method,
            pos_label=pos_label,
        )

    def _get_predictions(
        self,
        *,
        data_source: Literal["train", "test"],
        response_method: str | list[str] | tuple[str, ...],
        pos_label: PositiveLabel | None = None,
    ) -> ArrayLike:
        """Get estimator's predictions, and adapt them to `pos_label` if needed.

        Internal helpers used by the metrics.

        Returns
        -------
        np.ndarray of shape (n_samples,) or (n_samples, n_classes)
            The predictions.
            The shape is (n_samples,) if:
                - response_method is "predict"
                - OR if pos_label is specified (binary-classification only)
            Otherwise it's (n_samples, n_classes)
        """
        if data_source not in ("train", "test"):
            raise ValueError(f"Invalid data source: {data_source}")
        if pos_label is not None and self.ml_task != "binary-classification":
            raise ValueError(f"Cannot specify a `pos_label` for task {self.ml_task}")

        method_name = _check_response_method(self.estimator_, response_method).__name__
        self.cache_predictions(response_methods=[method_name], data_source=data_source)
        predictions = self._predictions[(data_source, method_name)]

        if method_name == "predict":
            return predictions
        # Adapt to pos_label / check shape if needed:
        if method_name in ("predict_proba", "predict_log_proba"):
            if pos_label is None:
                return predictions
            # Adapt to pos_label:
            return _process_predict_proba(
                y_pred=predictions,
                target_type="binary",
                classes=self.estimator_.classes_,
                pos_label=pos_label,
            )
        elif method_name == "decision_function":
            if pos_label is not None:
                # Adapt to pos_label:
                return _process_decision_function(
                    y_pred=predictions,
                    target_type="binary",
                    classes=self.estimator_.classes_,
                    pos_label=pos_label,
                )

            if self.ml_task == "multiclass-classification":
                _, d = predictions.shape
                if d != len(self.estimator_.classes_):
                    raise ValueError(f"Unexpected decision function shape[1]: {d}")
                return predictions
            # scikit-learn returns a (n_samples,) array that corresponds to classes[-1]
            # we normalize to a (n_samples, 2) shape with similar semantic than
            # predict_proba
            return np.vstack((-predictions, predictions)).T
        else:
            raise ValueError(f"Unexpected response_method: {method_name}")

    @property
    def ml_task(self):
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

    @property
    def y_test(self) -> ArrayLike | None:
        return self._y_test

    @property
    def pos_label(self) -> PositiveLabel | None:
        return self._pos_label

    @property
    def estimator_name_(self) -> str:
        if isinstance(self._estimator, Pipeline):
            name = self._estimator[-1].__class__.__name__
        else:
            name = self._estimator.__class__.__name__
        return name

    @property
    def fit(self) -> str | bool:
        return self._fit

    ####################################################################################
    # Methods related to the help and repr
    ####################################################################################

    def _get_help_title(self) -> str:
        return f"Tools to diagnose estimator {self.estimator_name_}"

    def _get_help_legend(self) -> str:
        return (
            "[cyan](↗︎)[/cyan] higher is better [orange1](↘︎)[/orange1] lower is better"
        )

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"""{self.__class__.__name__}:
        {self.estimator_!r}

        {self.metrics.summarize().frame()}"""

    def _html_repr_fragments(self) -> dict[str, str]:
        """HTML snippets for the report body (metrics, estimator diagram, data table).

        Used by :meth:`_repr_html_` and by :class:`~skore.ComparisonReport` to embed
        one report's views in the comparison HTML repr.
        """
        match self.X_train, self.X_test:
            case None, None:
                data_source = None
            case _, None:
                data_source = "train"
            case None, _:
                data_source = "test"
            case _:
                data_source = "both"
        if data_source is None:
            table_report_html = "<p>No data provided</p>"
            metrics_html = "<p>No data provided</p>"
        else:
            table_report = skrub.TableReport(
                self.data._prepare_dataframe_for_display(
                    data_source=data_source,
                    with_y=True,
                    subsample=None,
                    subsample_strategy="head",
                    seed=None,
                ),
                max_plot_columns=0,
                max_association_columns=0,
                verbose=False,
            )
            table_report._set_minimal_mode()
            table_report_html = table_report.html_snippet()
            metrics_html = (
                self.metrics.summarize(
                    data_source="train" if data_source == "train" else "test"
                )
                .frame()
                ._repr_html_()
            )
        try:
            estimator_html = self.estimator_._repr_html_()
        except Exception:
            estimator_html = f"<p>{html.escape(repr(self.estimator_))}</p>"

        return {
            "metrics_summary": metrics_html,
            "estimator_display": estimator_html,
            "table_report": table_report_html,
        }

    def _repr_html_(self) -> str:
        """HTML representation of estimator.

        This is redundant with the logic of `_repr_mimebundle_`. The latter
        should be favored in the long term, `_repr_html_` is only
        implemented for consumers who do not interpret `_repr_mimbundle_`.
        """
        fragments = self._html_repr_fragments()
        container_id = f"skore-estimator-report-{uuid.uuid4().hex[:8]}"
        help_doc_url = get_documentation_url(obj=self, method_name="help")
        report_class_name = self.__class__.__name__
        metrics_accessor_doc_url = get_documentation_url(
            obj=self, accessor_name="metrics"
        )
        inspection_accessor_doc_url = get_documentation_url(
            obj=self, accessor_name="inspection"
        )
        data_accessor_doc_url = get_documentation_url(obj=self, accessor_name="data")
        return render_template(
            "estimator_report.html.j2",
            {
                "container_id": container_id,
                "help_doc_url": help_doc_url,
                "report_class_name": report_class_name,
                "metrics_accessor_doc_url": metrics_accessor_doc_url,
                "inspection_accessor_doc_url": inspection_accessor_doc_url,
                "data_accessor_doc_url": data_accessor_doc_url,
                **fragments,
            },
        )

    def _repr_mimebundle_(self, **kwargs):
        """Mime bundle used by jupyter kernels to display estimator."""
        output = {"text/plain": repr(self)}
        output["text/html"] = self._repr_html_()
        return output

    ####################################################################################
    # Methods related to the cache
    ####################################################################################

    def _read_cache(
        self,
        accessor_name: str,
        method_name: str,
        data_source: DataSource | None,
        kwargs: dict[str, Any] | None,
        *,
        result_only: bool = True,
    ) -> Any:
        if kwargs is not None and kwargs.get("seed", 0) is None:
            return None
        key = joblib.hash((accessor_name, method_name, data_source, kwargs or {}))
        value = self._cache.get(key)
        if value is None:
            return None
        if result_only:
            *_, result = value
            return result
        return value

    def _write_cache(
        self,
        accessor_name: str,
        method_name: str,
        data_source: DataSource | None,
        kwargs: dict[str, Any],
        result: Any,
    ) -> None:
        key = joblib.hash((accessor_name, method_name, data_source, kwargs))
        value = (
            accessor_name,
            method_name,
            data_source,
            kwargs,
            result,
        )
        self._cache[key] = value

    def _get_cached_results(
        self,
        accessor_name: str,
        method_name: str | None = None,
        data_source: DataSource | None = None,
    ) -> list[tuple[str, str, DataSource | None, dict[str, Any], Any]]:
        """Return accessor-level cached results stored in ``self._cache``.

        This helper exposes the inspectable part of the estimator cache used by
        accessors such as ``metrics`` and ``inspection``. Prediction arrays and
        prediction timings are stored separately in ``self._predictions`` and are not
        returned by this method.

        Notes
        -----
        ``self._cache`` is a mapping from a hashed cache key to a tuple with the
        following structure::

            (
                accessor_name: str,
                method_name: str,
                data_source: {"train", "test"} | None,
                kwargs: dict[str, Any],
                result: Any,
            )

        The dictionary key is an implementation detail. Callers that need to inspect
        cached results should rely on this method instead of reading ``self._cache``
        directly.

        Parameters
        ----------
        accessor_name : str
            Name of the accessor that produced the cached result, for example
            ``"metrics"`` or ``"inspection"``.

        method_name : str or None, default=None
            Restrict results to a single accessor method. If ``None``, return cached
            results for all methods of the accessor.

        data_source : {"train", "test"} or None, default=None
            Restrict results to a single data source. If ``None``, do not filter on
            the data source.

        Returns
        -------
        cached_results : list of tuple
            Cached entries matching the requested filters, each represented as
            ``(accessor_name, method_name, data_source, kwargs, result)``.
        """
        # This API is used by projects to inspect computed results
        # changing it might break `skore-*-project`s

        def value_match(cached_value):
            return (
                cached_value[0] == accessor_name
                and (method_name is None or cached_value[1] == method_name)
                and (data_source is None or cached_value[2] == data_source)
            )

        return list(filter(value_match, self._cache.values()))
