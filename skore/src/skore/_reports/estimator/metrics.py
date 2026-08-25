from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if

from skore._displays.metrics.metrics_summary_display import (
    MetricsSummaryDisplay,
    MetricsSummaryRow,
)
from skore._externals.pandas_accessors import DirNamesMixin
from skore._metrics import (
    FitTime,
    Metric,
    MetricRow,
    MissingKwargsError,
    PredictTime,
    Score,
)
from skore._reports.base import BaseMetricsAccessor
from skore._sklearn.types import DataSource
from skore._utils.accessor import _check_supported_ml_task
from skore._utils.cache_key import make_cache_key

if TYPE_CHECKING:
    from skore._displays.metrics.confusion_matrix import ConfusionMatrixDisplay
    from skore._displays.metrics.precision_recall_curve import (
        PrecisionRecallCurveDisplay,
    )
    from skore._displays.metrics.prediction_error import PredictionErrorDisplay
    from skore._displays.metrics.roc_curve import RocCurveDisplay
    from skore._metrics import MetricLike
    from skore._reports.estimator.report import EstimatorReport  # noqa: F401


class _MetricsAccessor(BaseMetricsAccessor["EstimatorReport"], DirNamesMixin):
    """Accessor for metrics-related operations.

    You can access this accessor using the `metrics` attribute.
    """

    def summarize(
        self,
        *,
        data_source: DataSource | Literal["both"] = "test",
        metric: str | list[str] | None = None,
    ) -> MetricsSummaryDisplay:
        """Report a set of metrics for our estimator.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "both" : use both the train and test sets to compute the metrics and
              present them side-by-side.

        metric : str or list of str or None, default=None
            The metrics to report, from the list of registered metrics. None means show
            all registered metrics. To add a custom metric, see :meth:`add`.
            Metrics added with a ``neg_`` prefix can also be retrieved without it
            (e.g. ``"neg_mean_absolute_percentage_error"`` instead of
            ``"mean_absolute_percentage_error"``).

        Returns
        -------
        :class:`MetricsSummaryDisplay`
            A display containing the statistics for the metrics.

        See Also
        --------
        MetricsSummaryDisplay.frame : Export the summary; wide single-column
            layouts return a named :class:`pandas.Series`.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=0.2, pos_label=1)
        >>> summary = report.metrics.summarize().frame(favorability=True)
        >>> summary[~summary.index.isin(["fit_time", "predict_time"])]
                     LogisticRegression favorability
        metric
        accuracy               0.94...         (↗︎)
        precision              0.98...         (↗︎)
        recall                 0.92...         (↗︎)
        roc_auc                0.99...         (↗︎)
        log_loss               0.11...         (↘︎)
        brier_score            0.03...         (↘︎)
        >>> # Using scikit-learn metrics
        >>> report.metrics.summarize(metric="log_loss").frame(favorability=True)
                  LogisticRegression favorability
        metric
        log_loss            0.11...         (↘︎)
        >>> summary = report.metrics.summarize(
        ...    data_source="both"
        ... ).frame(favorability=True)
        >>> summary[~summary.index.isin(["fit_time", "predict_time"])]
                     LogisticRegression (train)  LogisticRegression (test) favorability
        metric
        accuracy                       0.96...                    0.94...         (↗︎)
        precision                      0.96...                    0.98...         (↗︎)
        recall                         0.97...                    0.92...         (↗︎)
        roc_auc                        0.99...                    0.99...         (↗︎)
        log_loss                       0.08...                    0.11...         (↘︎)
        brier_score                    0.02...                    0.03...         (↘︎)
        """
        if data_source == "both":
            train_summary = self._summarize_display(data_source="train", metric=metric)
            test_summary = self._summarize_display(data_source="test", metric=metric)

            combined = pd.concat(
                [train_summary.summary, test_summary.summary], ignore_index=True
            )
            return MetricsSummaryDisplay(
                summary=combined,
                report_type="estimator",
                errors=train_summary.errors + test_summary.errors,
            )

        return self._summarize_display(data_source=data_source, metric=metric)

    def _summarize_display(
        self,
        *,
        data_source: DataSource | Literal["both"],
        metric: str | list[str] | None = None,
    ) -> MetricsSummaryDisplay:
        if data_source == "both":
            train_summary = self._summarize_display(data_source="train", metric=metric)
            test_summary = self._summarize_display(data_source="test", metric=metric)

            combined = pd.concat(
                [train_summary.summary, test_summary.summary], ignore_index=True
            )
            return MetricsSummaryDisplay(
                summary=combined,
                report_type="estimator",
                errors=train_summary.errors + test_summary.errors,
            )

        registry = self._parent._metric_registry
        if isinstance(metric, str):
            parsed_metrics = [registry[metric]]
        elif isinstance(metric, Iterable) and metric:
            parsed_metrics = [registry[m] for m in metric]
        else:
            predictor = self._parent.estimator_
            if isinstance(predictor, Pipeline):
                predictor = predictor.steps[-1][1]
            has_default_score = getattr(type(predictor), "score", None) in (
                ClassifierMixin.score,
                RegressorMixin.score,
            )
            if has_default_score:
                parsed_metrics = [s for s in registry.values() if s.name != "score"]
            else:
                parsed_metrics = list(registry.values())

        rows: list[MetricsSummaryRow] = []
        errors = []
        for parsed_metric in parsed_metrics:
            try:
                metric_rows = parsed_metric.rows(
                    report=self._parent,
                    data_source=data_source,
                    **parsed_metric.kwargs,
                )
            except Exception as exception:
                metric_rows = [
                    MetricRow(
                        metric_verbose_name=parsed_metric.verbose_name,
                        greater_is_better=parsed_metric.greater_is_better,
                        label=None,
                        average=None,
                        output=None,
                        score=float("nan"),
                    )
                ]
                errors.append((parsed_metric, exception))

            rows.extend(
                {
                    "name": parsed_metric.name,
                    "verbose_name": row["metric_verbose_name"],
                    "estimator": self._parent.estimator_name_,
                    "data_source": data_source,
                    "greater_is_better": row["greater_is_better"],
                    "score": row["score"],
                    "label": row["label"],
                    "average": row["average"],
                    "output": row["output"],
                }
                for row in metric_rows
            )

        return MetricsSummaryDisplay._compute_data_for_display(
            rows, report_type="estimator", errors=errors
        )

    def _metric(
        self,
        metric_name: str,
        *,
        data_source: DataSource,
        **kwargs: Any,
    ) -> MetricsSummaryDisplay:
        """Compute a single metric, forwarding ``kwargs`` to the score function."""
        metric = self._parent._metric_registry[metric_name]
        rows = [
            cast(
                MetricsSummaryRow,
                {
                    "name": metric.name,
                    "verbose_name": row["metric_verbose_name"],
                    "estimator": self._parent.estimator_name_,
                    "data_source": data_source,
                    "greater_is_better": row["greater_is_better"],
                    "score": row["score"],
                    "label": row["label"],
                    "average": row["average"],
                    "output": row["output"],
                },
            )
            for row in metric.rows(
                report=self._parent, data_source=data_source, **kwargs
            )
        ]
        return MetricsSummaryDisplay._compute_data_for_display(
            rows, report_type="estimator", errors=[]
        )

    def available(self) -> list[str]:
        """List available metric names in the registry.

        Returns
        -------
        list[str]
            The list of available metric names.
        """
        return list(self._parent._metric_registry)

    def _resolve_metric(self, name: str) -> Metric | None:
        """Return the :class:`~skore._metrics.Metric` for ``name``, or None."""
        return self._parent._metric_registry.get(name)

    def add(
        self,
        metric: MetricLike,
        *,
        name: str | None = None,
        verbose_name: str | None = None,
        greater_is_better: bool = True,
        position: Literal["first", "last"] = "first",
        **kwargs: Any,
    ) -> None:
        """Add a custom metric to :meth:`summarize`.

        Parameters
        ----------
        metric : str, sklearn scorer, or callable
            The metric to add.

            - If a string, it will be run through :func:`sklearn.metrics.get_scorer`.
              Metrics that require a ``neg_`` prefix (e.g. ``"neg_mean_squared_error"``)
              may also be passed without it (e.g. ``"mean_squared_error"``); the alias
              is resolved automatically.
            - If a callable, it must have the signature
              ``(estimator, X, y_true, **kw) -> float``. It may also return a ``dict``
              mapping class labels to floats (e.g. ``{0: 0.9, 1: 0.85}``), in which case
              :meth:`summarize` will show one row per class label under the metric name.
              If your metric has the form ``(y_true, y_pred, **kw) -> float``, see
              :func:`sklearn.metrics.make_scorer` to convert it to a scorer.

        name : str or None, default=None
            Custom name for the metric. If ``None``, the name is inferred
            from the metric (e.g. the function's ``__name__``).

        verbose_name : str or None, default=None
            Custom verbose name for the metric which will be used for display
            purposes. If ``None``, the verbose name is inferred from the metric
            name.

        greater_is_better : bool, default=True
            Whether higher values are better (only for callables).

        position : {"first", "last"}, default="first"
            Where to place the metric in default :meth:`summarize` ordering.
            ``"first"`` inserts at the front; repeated ``"first"`` adds stack
            newest-first. ``"last"`` appends at the end.

        **kwargs : Any
            Default keyword arguments passed to the score function at call
            time. Only used when *metric* is a plain callable.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.metrics import make_scorer, mean_absolute_error
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, pos_label=1)
        >>> report.metrics.add(
        ...     make_scorer(mean_absolute_error, response_method="predict")
        ... )
        >>> report.metrics.summarize(metric="mean_absolute_error").frame(
        ...     verbose_name=True, flat_index=False
        ... )
        Metric
        Mean Absolute Error    0.05...
        Name: LogisticRegression, dtype: float64
        >>> report.metrics.mean_absolute_error()
        0.05...
        """
        try:
            self._parent._metric_registry.add(
                Metric.new(
                    metric,
                    name=name,
                    verbose_name=verbose_name,
                    greater_is_better=greater_is_better,
                    kwargs=kwargs,
                ),
                position=position,
            )
        except MissingKwargsError as error:
            args_msg = ", ".join(f"{kwarg}=..." for kwarg in error.missing_kwargs)
            raise ValueError(
                f"{error.msg} Pass those kwargs to add: add({error.metric}, {args_msg})"
            ) from error

    def remove(self, name: str) -> None:
        """Remove a metric from the registry.

        Parameters
        ----------
        name : str
            The name of the metric to remove.
        """
        # `remove` takes the report as input so that the MetricRegistry does not
        # need to have the report as an attribute, which would be a circular reference
        self._parent._metric_registry.remove(report=self._parent, name=name)

    def get(
        self,
        name: str,
        data_source: DataSource = "test",
        **kwargs,
    ) -> Any:
        """Get a metric value.

        Parameters
        ----------
        name : str
            Name of the metric to compute. Get all available metrics with
            :meth:`~EstimatorReport.metrics.available()`.
            Metrics added with a ``neg_`` prefix can also be retrieved
            without it; the alias is resolved automatically.
            When ``name`` is a valid Python identifier, the same value is also
            available as ``report.metrics.<name>(...)``.

        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        Returns
        -------
        Any
            The metric value in a human-readable shape: a scalar for
            single-output metrics, a mapping from class labels for per-class
            classification metrics, or an array for multioutput regression.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=0.2)
        >>> report.metrics.get("precision")
        {0: 0.90..., 1: 0.98...}
        >>> report.metrics.precision()
        {0: 0.90..., 1: 0.98...}
        """
        metric = self._parent._metric_registry[name]
        return metric.pretty(report=self._parent, data_source=data_source, **kwargs)

    def fit_time(self, *, cast: bool = True) -> float | None:
        """Get time to fit the estimator.

        Parameters
        ----------
        cast : bool, default=True
            Whether to cast the return value to a float. If `False`, the return value
            is `None` when the estimator is not fitted.

        Returns
        -------
        float or None
            The fit time in seconds, or `None` when not available.
        """
        return FitTime().pretty(report=self._parent, cast=cast)

    def predict_time(
        self,
        *,
        data_source: DataSource = "test",
        cast: bool = True,
    ) -> float | None:
        """Get prediction time if it has been already measured.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source for which the prediction time was recorded.

        cast : bool, default=True
            Whether to cast the numbers to floats. If `False`, the return value
            is `None` when the predictions have never been computed.

        Returns
        -------
        float or None
            The prediction time in seconds, or `None` when not available.
        """
        return PredictTime().pretty(
            report=self._parent, data_source=data_source, cast=cast
        )

    def timings(self) -> dict:
        """Get all measured processing times related to the estimator.

        When an estimator is fitted inside the :class:`~skore.EstimatorReport`, the time
        to fit is recorded. Prediction time is recorded when the estimator's
        `predict` method is computed and cached for a given data source. This function
        returns all the recorded times.

        Returns
        -------
        timings : dict
            The recorded times, in seconds,
            in the form of a `dict` with some or all of the following keys:

            - "fit_time", for the time to fit the estimator in the train set.
            - "predict_time_{data_source}", where data_source is "train" or "test"
              for the time to compute the predictions on the given data source.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = make_classification(random_state=42)
        >>> estimator = LogisticRegression()
        >>> report = evaluate(estimator, X, y, splitter=0.2)
        >>> report.metrics.timings()
        {'fit_time': ..., 'predict_time_test': ...}
        """
        times = {
            "fit_time": self.fit_time(cast=False),
            "predict_time_train": self.predict_time(data_source="train", cast=False),
            "predict_time_test": self.predict_time(data_source="test", cast=False),
        }
        return {k: v for k, v in times.items() if v is not None}

    @available_if(lambda self: Score.available(self._parent))
    def score(
        self,
        *,
        data_source: DataSource = "test",
    ) -> Any:
        """Compute the estimator's default score.

        This calls the underlying estimator's ``score`` method on the chosen data
        source. For :class:`skrub.DataOp` estimators, scorings registered via
        :meth:`~skrub.DataOp.skb.with_scoring` are used.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        Returns
        -------
        The default score of the estimator.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=0.2)
        >>> report.metrics.score()
        0.94...
        """
        return Score().pretty(report=self._parent, data_source=data_source)

    ####################################################################################
    # Methods related to displays
    ####################################################################################

    def _get_display(
        self,
        *,
        data_source: DataSource | Literal["both"],
        response_method: str | list[str] | tuple[str, ...],
        display_class: type[
            RocCurveDisplay
            | PrecisionRecallCurveDisplay
            | PredictionErrorDisplay
            | ConfusionMatrixDisplay
        ],
        display_kwargs: dict[str, Any],
    ) -> (
        RocCurveDisplay
        | PrecisionRecallCurveDisplay
        | PredictionErrorDisplay
        | ConfusionMatrixDisplay
    ):
        """Get the display from the cache or compute it.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "both" : use both the train set and the test set to compute the metric.

        response_method : str, list of str or tuple of str
            The response method.

        display_class : class
            The display class.

        display_kwargs : dict
            The display kwargs used by `display_class._compute_data_for_display`.

        Returns
        -------
        display : display_class
            The display.
        """
        if data_source == "both":
            displays = [
                self._get_display(
                    data_source=cast(DataSource, ds),
                    response_method=response_method,
                    display_class=display_class,
                    display_kwargs=display_kwargs,
                )
                for ds in ["train", "test"]
            ]
            return display_class._concatenate(
                displays,  # type: ignore[arg-type]
                report_type=self._parent._report_type,
                data_source=data_source,
            )

        # Compute cache key
        if "seed" in display_kwargs and display_kwargs["seed"] is None:
            cache_key = None
        else:
            cache_key = make_cache_key(
                "metrics", data_source, display_class.__name__, display_kwargs
            )

        cache_value = self._parent._cache.get(cache_key)
        if cache_value is not None:
            return cache_value

        _, y_true = self._parent._get_data_and_y_true(data_source=data_source)

        y_pred = self._parent._get_predictions(
            data_source=data_source, response_method=response_method
        )

        display = display_class._compute_data_for_display(
            y_true=y_true,
            y_pred=y_pred,
            report_type=self._parent._report_type,
            estimator=self._parent.estimator_,
            estimator_name=self._parent.estimator_name_,
            ml_task=self._parent._ml_task,
            data_source=data_source,
            **display_kwargs,
        )

        if cache_key is not None:
            # Unless seed is an int (i.e. the call is deterministic),
            # we do not cache
            self._parent._cache[cache_key] = display

        return display

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def roc(
        self,
        *,
        data_source: DataSource | Literal["both"] = "test",
    ) -> RocCurveDisplay:
        """Plot the ROC curve.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "both" : use both the train and test sets to compute the metrics and
              present them side-by-side.

        Returns
        -------
        :class:`RocCurveDisplay`
            The ROC curve display.

        See Also
        --------
        :class:`RocCurveDisplay` : Display class for ROC curve plots.

        Notes
        -----
        To keep the stored display lightweight, the ROC curve is downsampled
        to at most 500 points per class. Sampling is performed by picking
        evenly-spaced indices on the sorted thresholds.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=0.2)
        >>> display = report.metrics.roc()
        >>> display.set_style(relplot_kwargs={"color": "tab:red"})
        >>> display.plot()
        """
        response_method = ("predict_proba", "decision_function")
        display_kwargs = {"report_pos_label": self._parent.pos_label}
        from skore._displays.metrics.roc_curve import RocCurveDisplay

        display = cast(
            RocCurveDisplay,
            self._get_display(
                data_source=data_source,
                response_method=response_method,
                display_class=RocCurveDisplay,
                display_kwargs=display_kwargs,
            ),
        )
        return display

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def precision_recall(
        self,
        *,
        data_source: DataSource = "test",
    ) -> PrecisionRecallCurveDisplay:
        """Plot the precision-recall curve.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        Returns
        -------
        :class:`PrecisionRecallCurveDisplay`
            The precision-recall curve display.

        See Also
        --------
        :class:`PrecisionRecallCurveDisplay`
            Display class for precision-recall curve plots.

        Notes
        -----
        To keep the stored display lightweight, the precision-recall curve is
        downsampled to at most 500 points per class. Sampling is performed by
        picking evenly-spaced indices on the sorted thresholds.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=0.2)
        >>> display = report.metrics.precision_recall()
        >>> display.set_style(relplot_kwargs={"color": "tab:red"})
        >>> display.plot()
        """
        response_method = ("predict_proba", "decision_function")
        display_kwargs = {"report_pos_label": self._parent.pos_label}
        from skore._displays.metrics.precision_recall_curve import (
            PrecisionRecallCurveDisplay,
        )

        display = cast(
            PrecisionRecallCurveDisplay,
            self._get_display(
                data_source=data_source,
                response_method=response_method,
                display_class=PrecisionRecallCurveDisplay,
                display_kwargs=display_kwargs,
            ),
        )
        return display

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["regression", "multioutput-regression"]
        )
    )
    def prediction_error(
        self,
        *,
        data_source: DataSource | Literal["both"] = "test",
        subsample: float | int | None = 1_000,
        seed: int | None = None,
    ) -> PredictionErrorDisplay:
        """Plot the prediction error of a regression model.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "both" : use both the train and test sets to compute the metrics and
              present them side-by-side.

        subsample : float, int or None, default=1_000
            Sampling the samples to be shown on the scatter plot. If `float`,
            it should be between 0 and 1 and represents the proportion of the
            original dataset. If `int`, it represents the number of samples
            display on the scatter plot. If `None`, no subsampling will be
            applied. by default, 1,000 samples or less will be displayed.

        seed : int, default=None
            The seed used to initialize the random number generator used for the
            subsampling.

        Returns
        -------
        :class:`PredictionErrorDisplay`
            The prediction error display.

        See Also
        --------
        :class:`PredictionErrorDisplay` : Display class for prediction error plots.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import evaluate
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> report = evaluate(regressor, X, y, splitter=0.2)
        >>> display = report.metrics.prediction_error()
        >>> display.set_style(perfect_model_kwargs={"color": "tab:red"})
        >>> display.plot()
        """
        display_kwargs = {"subsample": subsample, "seed": seed}
        from skore._displays.metrics.prediction_error import PredictionErrorDisplay

        display = cast(
            PredictionErrorDisplay,
            self._get_display(
                data_source=data_source,
                response_method="predict",
                display_class=PredictionErrorDisplay,
                display_kwargs=display_kwargs,
            ),
        )
        return display

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def confusion_matrix(
        self,
        *,
        data_source: DataSource = "test",
    ) -> ConfusionMatrixDisplay:
        """Plot the confusion matrix.

        The confusion matrix shows the counts of correct and incorrect classifications
        for each class.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        Returns
        -------
        :class:`ConfusionMatrixDisplay`
            The confusion matrix display.

        See Also
        --------
        :class:`ConfusionMatrixDisplay` : Display class for confusion matrix plots.

        Notes
        -----
        To keep the stored display lightweight, the thresholded confusion matrices are
        downsampled to at most 500 points per class. Sampling is performed by picking
        evenly-spaced indices on the sorted thresholds.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=0.2)
        >>> display = report.metrics.confusion_matrix()
        >>> display.plot()

        With specific threshold for binary classification:

        >>> display = report.metrics.confusion_matrix()
        >>> display.plot(threshold_value=0.7, label=1)
        """
        if data_source == "both":
            raise ValueError(
                "data_source='both' is not supported for confusion_matrix."
            )

        if hasattr(self._parent.learner_, "predict_proba") or hasattr(
            self._parent.learner_, "decision_function"
        ):
            y_scores = self._parent._get_predictions(
                data_source=data_source,
                response_method=("predict_proba", "decision_function"),
            )
        else:
            y_scores = None

        display_kwargs: dict = {
            "report_pos_label": self._parent.pos_label,
            "y_scores": y_scores,
        }
        from skore._displays.metrics.confusion_matrix import ConfusionMatrixDisplay

        display = cast(
            ConfusionMatrixDisplay,
            self._get_display(
                data_source=data_source,
                response_method="predict",
                display_class=ConfusionMatrixDisplay,
                display_kwargs=display_kwargs,
            ),
        )
        return display
