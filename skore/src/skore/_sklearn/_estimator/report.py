from __future__ import annotations

import copy
import html
import uuid
import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import skrub
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils._response import (
    _check_response_method,
)
from sklearn.utils.validation import _num_samples, check_is_fitted

from skore._externals._pandas_accessors import DirNamesMixin
from skore._externals._sklearn_compat import _safe_indexing, is_clusterer
from skore._sklearn._base import _BaseReport
from skore._sklearn.find_ml_task import _find_ml_task
from skore._sklearn.metrics import MetricRegistry
from skore._sklearn.types import DataSource, PositiveLabel
from skore._utils._cache import Cache
from skore._utils._cache_key import make_cache_key
from skore._utils._measure_time import MeasureTime
from skore._utils._skrub import eval_X_y, is_skrub_learner, to_estimator, to_learner
from skore._utils.repr.data import get_documentation_url
from skore._utils.repr.html_repr import render_template

if TYPE_CHECKING:
    from skore._sklearn._diagnostic.accessor import _DiagnosisAccessor
    from skore._sklearn._estimator.data_accessor import _DataAccessor
    from skore._sklearn._estimator.inspection_accessor import _InspectionAccessor
    from skore._sklearn._estimator.metrics_accessor import _MetricsAccessor


_STATE_VERSION = 1


def _check_estimator_and_data(
    estimator, X_train, y_train, X_test, y_test, train_data, test_data
):
    if is_skrub_learner(estimator):
        initialized_with_data_op = True
        if any(v is not None for v in (X_train, y_train, X_test, y_test)):
            raise TypeError(
                "X_train, y_train, X_test, y_test cannot be provided when "
                "estimator is a SkrubLearner. "
                "Provide train_data and test_data instead."
            )
        test_data = (
            None if test_data is None else eval_X_y(estimator.data_op, test_data)
        )
        train_data = (
            None if train_data is None else eval_X_y(estimator.data_op, train_data)
        )
    else:
        initialized_with_data_op = False
        if train_data is not None or test_data is not None:
            raise TypeError(
                "train_data and test_data can only be provided when estimator "
                "is a SkrubLearner. "
                "Provide X_train, y_train, X_test, y_test instead."
            )
        estimator = to_learner(estimator)
        test_data = None if X_test is None else {"_skrub_X": X_test, "_skrub_y": y_test}
        train_data = (
            None if X_train is None else {"_skrub_X": X_train, "_skrub_y": y_train}
        )
    return initialized_with_data_op, estimator, train_data, test_data


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
        "diagnosis": {"name": "diagnosis"},
    }

    _report_type: Literal["estimator"] = "estimator"

    metrics: _MetricsAccessor
    inspection: _InspectionAccessor
    data: _DataAccessor
    diagnosis: _DiagnosisAccessor

    def _fit_estimator(
        self,
        estimator: BaseEstimator,
        data,
    ) -> tuple[BaseEstimator, float]:
        if data is None:
            raise ValueError(
                "The training data is required to fit the estimator. "
                "Please provide training data or a fitted estimator."
            )
        estimator_ = clone(estimator)
        with MeasureTime() as fit_time:
            estimator_.fit(data)
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
        estimator: BaseEstimator | skrub.DataOp,
        *,
        fit: Literal["auto"] | bool = "auto",
        X_train: ArrayLike | None = None,
        y_train: ArrayLike | None = None,
        X_test: ArrayLike | None = None,
        y_test: ArrayLike | None = None,
        train_data: dict | None = None,
        test_data: dict | None = None,
        pos_label: PositiveLabel | None = None,
    ) -> None:
        super().__init__()
        estimator = self._copy_estimator(estimator)
        self._raw_estimator = estimator
        self._fit = fit

        if isinstance(estimator, skrub.DataOp):
            if test_data is None and train_data is None:
                split = estimator.skb.train_test_split()
                test_data = split["test"]
                train_data = split["train"]
            estimator = estimator.skb.make_learner()
        if is_clusterer(estimator):
            raise ValueError(
                "Clustering models are not supported yet. Please use a"
                " classification or regression model instead."
            )

        self._initialized_with_data_op, estimator, self._train_data, self._test_data = (
            _check_estimator_and_data(
                estimator, X_train, y_train, X_test, y_test, train_data, test_data
            )
        )
        self._fit_time: float | None = None
        if fit == "auto":
            try:
                check_is_fitted(estimator)
                self._estimator = estimator
            except NotFittedError:
                self._estimator, self._fit_time = self._fit_estimator(
                    estimator, self._train_data
                )
        elif fit is True:
            self._estimator, self._fit_time = self._fit_estimator(
                estimator, self._train_data
            )
        else:  # fit is False
            self._estimator = estimator

        self._pos_label = pos_label
        self.fit_time_ = self._fit_time
        self._ml_task = _find_ml_task(self.y_test, estimator=self.estimator_)
        self._cache = Cache()
        # NOTE: Reports are immutable so we don't need cache invalidation

        self._metric_registry = MetricRegistry(self)

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

    def get_state(self) -> dict[str, Any]:
        """Return a serializable representation of the report state.

        This state is meant to ease serialization/deserialization of
        reports while preserving some backward compatibility across skore
        versions. In particular, this is more stable than pickling a report
        object directly, which can break when internal implementations change.
        """
        # split the cache between predictions and results:
        pred_key_names = {
            "predict",
            "predict_time",
            "decision_function",
            "predict_proba",
            "predict_log_proba",
        }

        predictions = {}
        cached_results = {}

        for key, val in self._cache.items():
            data_source, name, kwargs = key
            if name in pred_key_names:
                assert kwargs is None
                predictions[(data_source, name)] = val
            else:
                cached_results[key] = val

        return {
            "version": _STATE_VERSION,
            # -------- CORE STATE ---------
            "metadata": self._metadata,
            "initialized_with_data_op": self._initialized_with_data_op,
            "raw_estimator": self._raw_estimator,
            "ml_task": self._ml_task,
            "fit": self._fit,
            "fit_time": self.fit_time_,
            "pos_label": self._pos_label,
            "estimator": self._estimator,
            "data": {
                "train_data": self._train_data,
                "test_data": self._test_data,
            },
            "predictions": predictions,
            "metric_registry": self._metric_registry,
            # ---------- OPTIONAL STATE ------------
            # this part is less structured and not crucial for reconstructing a report
            # so we won't try ensuring backward compatibility.
            "optional": {
                "cache": cached_results,
            },
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> EstimatorReport:
        """Rebuild a report from :meth:`get_state` output."""
        version = state.get("version")
        if version != _STATE_VERSION:
            # in the future, we could support some BW compatibility instead of crashing
            raise ValueError(f"Unexpected state version: {version!r}")

        report_type = state["metadata"]["report_type"]
        if report_type != cls._report_type:
            raise ValueError(f"Unexpected report_type in state: {report_type}")

        report = cls.__new__(cls)

        report._metadata = state["metadata"]
        report._initialized_with_data_op = state["initialized_with_data_op"]
        report._ml_task = state["ml_task"]
        report._fit = state["fit"]
        report.fit_time_ = state["fit_time"]
        report._pos_label = state["pos_label"]
        report._estimator = state["estimator"]
        report._raw_estimator = state["raw_estimator"]
        data = state["data"]
        report._train_data = data["train_data"]
        report._test_data = data["test_data"]
        report._cache = Cache()
        report._cache.update(state["optional"]["cache"])
        report._cache.update(
            {
                make_cache_key(data_source, name): val
                for (data_source, name), val in state["predictions"].items()
            }
        )
        report._metric_registry = state["metric_registry"]

        return report

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

    def cache_predictions(
        self,
        data_source: DataSource | Literal["both"] = "both",
    ) -> None:
        """Cache estimator's predictions.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="both"
            The data source(s) for which to precompute predictions.

            - "test" : cache predictions for the test set only.
            - "train" : cache predictions for the train set only.
            - "both" : cache predictions for both train and test sets when available.

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
        if data_source == "both":
            self.cache_predictions(data_source="test")
            if self.X_train is not None:
                self.cache_predictions(data_source="train")
            return

        data = self._test_data if data_source == "test" else self._train_data
        if data is None:
            raise ValueError(
                f"No {data_source} features (i.e. X_{data_source}) were provided "
                f"when creating the report. Please provide the {data_source} "
                "features when creating the report."
            )

        pred_key = make_cache_key(data_source, "predict")
        time_key = make_cache_key(data_source, "predict_time")

        if pred_key in self._cache:
            return

        # This is for cases where `predict` cannot be inferred reliably
        # from decision_function/predict_proba:
        if not self._can_skip_predict:
            with MeasureTime() as pred_time:
                self._cache[pred_key] = self._estimator.predict(data)
            self._cache[time_key] = pred_time()

        has_proba = hasattr(self._estimator, "predict_proba")
        has_decision = hasattr(self._estimator, "decision_function")

        if not (has_proba or has_decision):
            return

        if has_decision:
            response, predictions, pred_time = (
                self._get_response_and_derived_predictions(
                    data, response_method="decision_function"
                )
            )
            decision_key = make_cache_key(data_source, "decision_function")
            self._cache[decision_key] = response
            if self._can_skip_predict:
                self._cache[time_key] = pred_time
                self._cache[pred_key] = predictions

        if has_proba:
            response, predictions, pred_time = (
                self._get_response_and_derived_predictions(
                    data, response_method="predict_proba"
                )
            )
            proba_key = make_cache_key(data_source, "predict_proba")
            self._cache[proba_key] = response
            log_key = make_cache_key(data_source, "predict_log_proba")
            # Most sklearn's estimator derive predict_log_proba this way
            # except for *NB models (naive bayes) that derive predict_proba
            # from predict_log_proba using exp:
            with np.errstate(divide="ignore"):
                self._cache[log_key] = np.log(response)
            if self._can_skip_predict:
                self._cache[time_key] = pred_time
                self._cache[pred_key] = predictions

    def _get_response_and_derived_predictions(self, data, response_method):
        """Compute a response array and derive class predictions.

        Returns
        -------
        response : ndarray of shape (n_samples, n_classes)
            For binary decision_function, the returned array is reshaped to
            (n_samples, 2) so it can be aligned with classes_.
        predictions : ndarray of shape (n_samples,) or None
            Predicted labels derived from response
            or None for ill-shaped decision function (OVO)
        pred_time : float
            Time spent computing ``response_method(data)`` in seconds.
        """
        with MeasureTime() as pred_time:
            response = getattr(self._estimator, response_method)(data)
        classes = to_estimator(self._estimator).classes_
        if response_method == "decision_function":
            if self.ml_task == "binary-classification":
                response = np.vstack((-response, response)).T
            if response.shape[1] != len(classes):
                return response, None, pred_time()
        predictions = classes[np.argmax(response, axis=1)]
        return response, predictions, pred_time()

    @cached_property
    def _can_skip_predict(self) -> bool:
        """Return whether `predict` can be inferred reliably.

        This probes a small sample of the available data and checks whether
        `predict(X)` matches the labels derived from `predict_proba(X)` or
        `decision_function(X)`. The result is cached because running the probe
        requires extra predictions.
        """
        estimator = to_estimator(self._estimator)
        if isinstance(estimator, MetaEstimatorMixin | Pipeline):
            return False

        response_methods = ["decision_function", "predict_proba"]
        try:
            method = _check_response_method(estimator, response_methods)
        except AttributeError:
            return False
        data = self.train_data if self.test_data is None else self.test_data
        assert data is not None

        # sample data for the probing:
        X = data["_skrub_X"]
        n_samples = _num_samples(X)
        sample_size = 100
        if n_samples <= sample_size:
            sampled_data = data
        else:
            rng = np.random.default_rng(0)
            indices = rng.choice(n_samples, size=sample_size, replace=False)
            X_sample = _safe_indexing(X, indices, axis=0)
            sampled_data = data | {"_skrub_X": X_sample}

        # probe:
        predictions = self._estimator.predict(sampled_data)
        _, deduced_predictions, _ = self._get_response_and_derived_predictions(
            sampled_data,
            response_method=method.__name__,
        )
        if deduced_predictions is None:
            return False
        return np.array_equal(predictions, deduced_predictions)

    def _get_data_and_y_true(
        self,
        *,
        data_source: DataSource,
    ) -> tuple[dict, ArrayLike]:
        """Get the requested dataset.

        Parameters
        ----------
        data_source : {"test", "train"}
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        Returns
        -------
        data : dict of input data
            The requested dataset.

        y : array-like of shape (n_samples,)
            The target labels.
        """
        if data_source not in ["train", "test"]:
            raise ValueError(
                f"Invalid data source: {data_source}. Possible values are: test, train."
            )
        if getattr(self, f"{data_source}_data") is None:
            raise ValueError(
                f"No {data_source} data were provided when creating the report."
            )
        if data_source == "test":
            assert self.test_data is not None
            assert self.y_test is not None
            return self.test_data, self.y_test
        assert data_source == "train"
        assert self.train_data is not None
        assert self.y_train is not None
        return self.train_data, self.y_train

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
        self.cache_predictions(data_source=data_source)
        cache_key = make_cache_key(data_source, method_name)
        predictions = self._cache[cache_key]

        if method_name == "predict":
            return predictions

        # check shape if needed:
        if (
            self.ml_task == "multiclass-classification"
            and method_name == "decision_function"
            and predictions.shape[1] != len(self.estimator_.classes_)
        ):
            raise ValueError(
                "Decision function output should have as many columns "
                f"as there are classes; expected {len(self.estimator_.classes_)} "
                f"but got {predictions.shape[1]}."
            )

        if pos_label is None:
            return predictions

        # Adapt to pos_label:
        # (copied from sklearn's _process_predict_proba)
        col_idx = np.flatnonzero(self.estimator_.classes_ == pos_label)[0]
        return predictions[:, col_idx]

    @property
    def ml_task(self):
        return self._ml_task

    @property
    def estimator(self) -> BaseEstimator:
        return self.estimator_

    @property
    def estimator_(self) -> BaseEstimator:
        if self._initialized_with_data_op:
            return self._estimator
        return to_estimator(self._estimator)

    @property
    def X_train(self) -> ArrayLike | None:
        return (self._train_data or {}).get("_skrub_X")

    @property
    def y_train(self) -> ArrayLike | None:
        return (self._train_data or {}).get("_skrub_y")

    @property
    def X_test(self) -> ArrayLike | None:
        return (self._test_data or {}).get("_skrub_X")

    @property
    def y_test(self) -> ArrayLike | None:
        return (self._test_data or {}).get("_skrub_y")

    @property
    def train_data(self) -> dict | None:
        return None if self._train_data is None else self._train_data.copy()

    @property
    def test_data(self) -> dict | None:
        return None if self._test_data is None else self._test_data.copy()

    @property
    def pos_label(self) -> PositiveLabel | None:
        return self._pos_label

    @property
    def fit(self) -> str | bool:
        return self._fit

    @property
    def estimator_name_(self) -> str:
        if isinstance(self._raw_estimator, Pipeline):
            name = self._raw_estimator[-1].__class__.__name__
        else:
            name = self._raw_estimator.__class__.__name__
        return name

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
                .reset_index()
                .to_html(index=False)
            )
        try:
            estimator_html = self.estimator_._repr_html_()
        except Exception:
            estimator_html = f"<p>{html.escape(repr(self.estimator_))}</p>"

        diagnostic = self.diagnosis()
        diagnostic_html = (
            "<div class='report-diagnostic-details'>"
            f"{len(diagnostic.frame(severity='issue'))} issue(s), "
            f"{len(diagnostic.frame(severity='tip'))} tip(s), "
            f"{len(diagnostic.frame(severity='passed'))} passed, "
            f"{diagnostic.n_ignored_codes} ignored."
            "</div>"
        )

        return {
            "metrics_summary": metrics_html,
            "estimator_display": estimator_html,
            "table_report": table_report_html,
            "diagnostic": diagnostic_html,
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
        diagnose_documentation_url = get_documentation_url(
            obj=self, method_name="diagnose"
        )
        return render_template(
            "estimator_report.html.j2",
            {
                "container_id": container_id,
                "help_doc_url": help_doc_url,
                "report_class_name": report_class_name,
                "report_title": f"Report for {self.estimator_name_}",
                "metrics_accessor_doc_url": metrics_accessor_doc_url,
                "inspection_accessor_doc_url": inspection_accessor_doc_url,
                "data_accessor_doc_url": data_accessor_doc_url,
                "diagnose_documentation_url": diagnose_documentation_url,
                **fragments,
            },
        )

    def _repr_mimebundle_(self, **kwargs):
        """Mime bundle used by jupyter kernels to display estimator."""
        output = {"text/plain": repr(self)}
        output["text/html"] = self._repr_html_()
        return output
