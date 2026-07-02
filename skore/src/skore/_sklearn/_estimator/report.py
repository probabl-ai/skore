from __future__ import annotations

import html
import uuid
from dataclasses import asdict
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

import numpy as np
import skrub
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils._response import (
    _check_response_method,
)
from sklearn.utils.validation import _num_samples, check_is_fitted
from skrub._reporting._summarize import summarize_dataframe

from skore._externals._pandas_accessors import DirNamesMixin
from skore._externals._sklearn_compat import _safe_indexing, is_clusterer
from skore._sklearn._base import _BaseReport
from skore._sklearn._checks.model_checks import _BUILTIN_CHECKS
from skore._sklearn.find_ml_task import _find_ml_task
from skore._sklearn.metrics import MetricRegistry
from skore._sklearn.types import DataSource, PositiveLabel
from skore._utils._cache import Cache
from skore._utils._cache_key import make_cache_key
from skore._utils._measure_time import MeasureTime
from skore._utils._skrub import eval_X_y, is_skrub_learner, to_estimator, to_learner
from skore._utils.repr.data import get_documentation_url
from skore._utils.repr.html_repr import render_template
from skore._utils.repr.markdown import markdown_data_section, report_markdown_context
from skore._utils.repr.utils import repair_estimator_html_for_slotted_host

if TYPE_CHECKING:
    from skore._sklearn._checks.accessor import _ChecksAccessor
    from skore._sklearn._estimator.data_accessor import _DataAccessor
    from skore._sklearn._estimator.inspection_accessor import _InspectionAccessor
    from skore._sklearn._estimator.metrics_accessor import _MetricsAccessor
    from skore._sklearn.types import EstimatorLike


_STATE_VERSION = 1


class PredictTime(TypedDict):
    train: NotRequired[float]
    test: NotRequired[float]


def _check_estimator_and_data(
    estimator: EstimatorLike,
    X_train: ArrayLike | None,
    y_train: ArrayLike | None,
    X_test: ArrayLike | None,
    y_test: ArrayLike | None,
    train_data: dict | None,
    test_data: dict | None,
) -> tuple[bool, skrub.SkrubLearner, dict | None, dict]:
    """Check and validate the estimator and data."""
    if is_skrub_learner(estimator):
        initialized_with_data_op = True
        if any(v is not None for v in (X_train, y_train, X_test, y_test)):
            raise TypeError(
                "X_train, y_train, X_test, y_test cannot be provided when "
                "estimator is a SkrubLearner. "
                "Provide train_data and test_data instead."
            )
        if test_data is None:
            raise TypeError(
                "test_data must be provided when estimator is a SkrubLearner."
            )
        test_data = eval_X_y(estimator.data_op, test_data)
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
        if X_test is None or y_test is None:
            raise TypeError(
                "Test data must be provided (unless estimator is a "
                "SkrubLearner and test_data is provided instead)."
            )
        test_data = {"_skrub_X": X_test, "_skrub_y": y_test}
        train_data = (
            None if X_train is None else {"_skrub_X": X_train, "_skrub_y": y_train}
        )
    return initialized_with_data_op, estimator, train_data, test_data


class EstimatorReport(_BaseReport, DirNamesMixin):
    """Provide tools to validate and inspect a fitted estimator.

    Refer to the :ref:`estimator_report` section of the user guide for more details.

    Parameters
    ----------
    estimator : estimator object
        Estimator to make the report from. An estimator can be one of the following:

        - a scikit-learn compatible estimator as a :class:`~sklearn.base.BaseEstimator`;
        - a skrub :class:`~skrub.DataOp` to preprocess the data;
        - a skrub :class:`~skrub.SkrubLearner` extracted from a :class:`~skrub.DataOp`
          by calling :meth:`~skrub.DataOp.skb.make_learner`.

        If the estimator is not fitted, it is cloned and then fitted on the training
        data.

    X_train : {array-like, sparse matrix} of shape (n_samples, n_features) or \
            None
        Training data.

    y_train : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        Training target.

    X_test : {array-like, sparse matrix} of shape (n_samples, n_features) or None
        Testing data. It should have the same structure as the training data.

    y_test : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        Testing target.

    train_data : dict or None
        When ``estimator`` is a skrub :class:`~skrub.SkrubLearner`, bindings for
        variables contained in the DataOp that was used to create this learner
        (e.g. ``{"X": X_df, "other_table": df, ...}``).

    test_data : dict or None
        When ``estimator`` is a skrub :class:`~skrub.SkrubLearner`, bindings for
        variables contained in the DataOp that was used to create this learner
        (e.g. ``{"X": X_df, "other_table": df, ...}``).

    pos_label : int, float, bool or str, default=None
        For binary classification, the positive class to use for metrics and displays
        that need one. If `None`, skore does not infer a default positive class.
        Binary metrics and displays that support it will expose all classes instead.
        This parameter is rejected for non-binary tasks.

    Attributes
    ----------
    estimator_ : estimator object
        The fitted estimator, exposed with the same interface as ``estimator``:

        - if the input was a regular scikit-learn estimator, its ``predict`` method
          should be used with arrays as usual, e.g. ``Report.estimator_.predict(X)``;
        - if the input was a :class:`skrub.DataOp` or a :class:`skrub.SkrubLearner`,
          ``estimator_`` is a :class:`skrub.SkrubLearner` so its ``predict`` method
          should be used with an environment dict, e.g.
          ``Report.estimator_.predict({"a": ..., "b": ...})``.

    estimator : estimator object
        The estimator that was given as input.

    learner_ : skrub.SkrubLearner
        The fitted estimator wrapped in a :class:`skrub.SkrubLearner`. If the
        input was already a :class:`skrub.SkrubLearner`, it is used as-is without
        further wrapping.

    estimator_name_ : str
        The name of the estimator.

    ml_task : str
        The machine learning task inferred from the data and estimator.

    metrics : MetricsAccessor
        Accessor for computing and plotting metrics.

    inspection : InspectionAccessor
        Accessor for model inspection (coefficients, feature importance, etc.).

    data : DataAccessor
        Accessor for dataset analysis.

    checks : ChecksAccessor
        Accessor for running diagnostic checks.

    See Also
    --------
    skore.CrossValidationReport
        Report of cross-validation results.

    skore.ComparisonReport
        Report of comparison between estimators.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> estimator = LogisticRegression()
    >>> from skore import EstimatorReport
    >>> report = EstimatorReport(
    ...     estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    ... )
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]] = {
        "data": {"name": "data"},
        "metrics": {"name": "metrics"},
        "inspection": {"name": "inspection"},
        "checks": {"name": "checks"},
    }

    _report_type: Literal["estimator"] = "estimator"

    metrics: _MetricsAccessor
    inspection: _InspectionAccessor
    data: _DataAccessor
    checks: _ChecksAccessor

    def _fit_estimator(
        self,
        estimator: EstimatorLike,
        data: dict | None,
    ) -> tuple[EstimatorLike, float]:
        """Clone then fit the estimator on the training data."""
        if data is None:
            raise ValueError(
                "The training data is required to fit the estimator. "
                "Please provide training data or a fitted estimator."
            )
        estimator_ = clone(estimator)
        with MeasureTime() as fit_time:
            estimator_.fit(data)
        return estimator_, fit_time()

    def __init__(
        self,
        estimator: EstimatorLike,
        *,
        X_train: ArrayLike | None = None,
        y_train: ArrayLike | None = None,
        X_test: ArrayLike | None = None,
        y_test: ArrayLike | None = None,
        train_data: dict | None = None,
        test_data: dict | None = None,
        pos_label: PositiveLabel | None = None,
    ) -> None:
        super().__init__()
        self.estimator = estimator

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
        try:
            check_is_fitted(estimator)
            self.learner_ = estimator
        except NotFittedError:
            self.learner_, self._fit_time = self._fit_estimator(
                estimator, self._train_data
            )

        self._pos_label = pos_label
        self._ml_task = _find_ml_task(self.y_test, estimator=self.estimator_)
        self._cache = Cache()
        # NOTE: Reports are immutable so we don't need cache invalidation

        self._predict_time: PredictTime = {}
        self._cache_predictions(data_source="test")

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

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the report state.

        This state is meant to ease serialization/deserialization of
        reports while preserving some backward compatibility across skore
        versions. In particular, this is more stable than pickling a report
        object directly, which can break when internal implementations change.
        """
        # split the cache between predictions and results
        pred_key_names = {
            "predict",
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
            "estimator": self.estimator,
            "ml_task": self._ml_task,
            "fit_time": self._fit_time,
            "predict_time": self._predict_time,
            "pos_label": self._pos_label,
            "learner": self.learner_,
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
    def from_dict(cls, state: dict[str, Any]) -> EstimatorReport:
        """Build a report from :meth:`to_dict` output."""
        version = state.get("version")
        if version != _STATE_VERSION:
            # For now, no backward compatibility
            raise ValueError(f"Unexpected state version: {version!r}")

        report_type = state["metadata"]["report_type"]
        if report_type != cls._report_type:
            raise ValueError(f"Unexpected report_type in state: {report_type}")

        report = cls.__new__(cls)

        report._metadata = state["metadata"]
        report._initialized_with_data_op = state["initialized_with_data_op"]
        report._ml_task = state["ml_task"]
        report._fit_time = state["fit_time"]
        report._predict_time = state["predict_time"]
        report._pos_label = state["pos_label"]
        report.learner_ = state["learner"]
        report.estimator = state["estimator"]
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
        report._checks_registry = list(_BUILTIN_CHECKS)

        return report

    def _clear_cache(self) -> None:
        """Clear the cache."""
        self._cache = Cache()

    def _cache_predictions(
        self,
        *,
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
        """
        if data_source == "both":
            self._cache_predictions(data_source="test")
            if self.X_train is not None:
                self._cache_predictions(data_source="train")
            return

        data = self._test_data if data_source == "test" else self._train_data
        if data is None:
            raise ValueError(
                f"No {data_source} features (i.e. X_{data_source}) were provided "
                f"when creating the report. Please provide the {data_source} "
                "features when creating the report."
            )

        pred_key = make_cache_key(data_source, "predict")
        if pred_key in self._cache:
            return

        # This is for cases where `predict` cannot be inferred reliably
        # from decision_function/predict_proba:
        if not self._can_skip_predict:
            with MeasureTime() as pred_time:
                self._cache[pred_key] = self.learner_.predict(data)
            self._predict_time[data_source] = pred_time()

        has_proba = hasattr(self.learner_, "predict_proba")
        has_decision = hasattr(self.learner_, "decision_function")

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
                self._predict_time[data_source] = pred_time
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
                self._predict_time[data_source] = pred_time
                self._cache[pred_key] = predictions

    def _get_response_and_derived_predictions(
        self,
        data: dict,
        response_method: Literal["predict", "predict_proba", "decision_function"],
    ) -> tuple[ArrayLike, ArrayLike | None, float]:
        """Compute a response array and derive class predictions.

        Parameters
        ----------
        data : dict
            The data to use to compute the response and derive predictions.
        response_method : str
            The response method to use to compute the response and derive predictions.
            Can be "decision_function" or "predict_proba".

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
            response = getattr(self.learner_, response_method)(data)
        classes = self.learner_.classes_
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
        response_methods = ["decision_function", "predict_proba"]
        try:
            method = _check_response_method(self.learner_, response_methods)
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
        predictions = self.learner_.predict(sampled_data)
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
        data_source : {"test", "train"}
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
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=0.2)
        >>> predictions = report.get_predictions(data_source="test")
        >>> predictions.shape
        (114,)
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
        self._cache_predictions(data_source=data_source)
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
    def estimator_(self) -> EstimatorLike:
        """The report's fitted estimator."""
        if self._initialized_with_data_op:
            return self.learner_
        return to_estimator(self.learner_)

    @property
    def X_train(self) -> ArrayLike | None:
        return (self._train_data or {}).get("_skrub_X")

    @property
    def y_train(self) -> ArrayLike | None:
        return (self._train_data or {}).get("_skrub_y")

    @property
    def X_test(self) -> ArrayLike:
        return self._test_data["_skrub_X"]

    @property
    def y_test(self) -> ArrayLike:
        return self._test_data["_skrub_y"]

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
    def estimator_name_(self) -> str:
        if isinstance(self.estimator, Pipeline):
            name = self.estimator[-1].__class__.__name__
        else:
            name = self.estimator.__class__.__name__
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
        metrics_frame = self.metrics.summarize(data_source="test").frame(
            format="auto", verbose_name=True
        )
        return f"""{self.__class__.__name__}:
        {self.estimator_name_!r}

        {metrics_frame}
        Call `report.to_markdown()` for a markdown summary of the report's contents."""

    def _html_repr_fragments(self) -> dict[str, str]:
        """HTML snippets for the report body (metrics, estimator diagram, data table).

        Used by :meth:`_repr_html_` and by :class:`~skore.ComparisonReport` to embed
        one report's views in the comparison HTML repr.
        """
        table_report = skrub.TableReport(
            self.data._prepare_dataframe_for_display(
                data_source="both" if self.X_train is not None else "test",
                with_y=True,
                subsample=None,
                subsample_strategy="head",
                seed=None,
            ),
            plot_distributions=False,
            verbose=False,
        )
        table_report._set_minimal_mode()
        table_report_html = table_report.html_snippet()
        metrics_html = (
            self.metrics.summarize(data_source="test")
            .frame(format="auto", verbose_name=True)
            .reset_index()
            .to_html(index=False)
        )
        try:
            estimator_html = repair_estimator_html_for_slotted_host(
                self.estimator_._repr_html_()
            )
        except Exception:
            estimator_html = f"<p>{html.escape(repr(self.estimator_))}</p>"

        checks_summary_html = self._checks_summary_html_fragment()

        return {
            "metrics_summary": metrics_html,
            "estimator_display": estimator_html,
            "table_report": table_report_html,
            "checks_summary": checks_summary_html,
        }

    def _repr_html_(self) -> str:
        """HTML representation of estimator.

        This is redundant with the logic of `_repr_mimebundle_`. The latter
        should be favored in the long term, `_repr_html_` is only
        implemented for consumers who do not interpret `_repr_mimbundle_`.
        """
        fragments = self._html_repr_fragments()
        container_id = f"skore-estimator-report-{uuid.uuid4().hex[:8]}"
        report_class_name = self.__class__.__name__
        metrics_accessor_doc_url = get_documentation_url(
            obj=self, accessor_name="metrics"
        )
        inspection_accessor_doc_url = get_documentation_url(
            obj=self, accessor_name="inspection"
        )
        data_accessor_doc_url = get_documentation_url(obj=self, accessor_name="data")
        checks_documentation_url = get_documentation_url(
            obj=self, accessor_name="checks"
        )
        help_ctx = asdict(self._build_help_data())
        help_ctx["is_report"] = True
        return render_template(
            "report/estimator_report.html.j2",
            {
                "container_id": container_id,
                "report_class_name": report_class_name,
                "report_title": f"Report for {self.estimator_name_}",
                "metrics_accessor_doc_url": metrics_accessor_doc_url,
                "inspection_accessor_doc_url": inspection_accessor_doc_url,
                "data_accessor_doc_url": data_accessor_doc_url,
                "checks_documentation_url": checks_documentation_url,
                **fragments,
                **help_ctx,
            },
        )

    def _repr_mimebundle_(self, **kwargs):
        """Mime bundle used by jupyter kernels to display estimator."""
        output = {"text/plain": repr(self)}
        output["text/html"] = self._repr_html_()
        return output

    def to_markdown(self) -> str:
        """Return a markdown summary of the report.

        The summary contains four sections (Estimator, Metrics, Checks, Data) that
        mirror the tabs of the HTML representation. Each section ends with a pointer
        to the corresponding accessor for full details.

        Returns
        -------
        str
            The markdown summary of the report.
        """
        metrics_text = repr(
            self.metrics.summarize(data_source="test").frame(
                format="auto", verbose_name=True
            )
        )
        timings = self.metrics.timings()
        summary = summarize_dataframe(
            self.data._prepare_dataframe_for_display(
                data_source="both" if self.X_train is not None else "test",
                with_y=True,
                subsample=None,
                subsample_strategy="head",
                seed=None,
            ),
            with_plots=False,
            with_associations=False,
            verbose=0,
        )
        return render_template(
            "report/estimator_report_markdown.j2",
            {
                **report_markdown_context(self),
                "fit_time": timings.get("fit_time"),
                "predict_time": timings.get("predict_time_test"),
                "metrics_text": metrics_text,
                **markdown_data_section(
                    summary,
                    data_label="full" if self.X_train is not None else "test",
                ),
            },
        )
