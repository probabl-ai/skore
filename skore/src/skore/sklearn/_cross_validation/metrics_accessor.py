from typing import Any, Literal, Optional, Union

from sklearn.utils.metaestimators import available_if

from skore.sklearn._base import _get_cached_response_values
from skore.sklearn._comparison.metrics_accessor import (
    _MetricsAccessor as _ComparisonMetricsAccessor,
)
from skore.sklearn._cross_validation.report import CrossValidationReport
from skore.sklearn._plot import (
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from skore.utils._accessor import _check_supported_ml_task
from skore.utils._progress_bar import progress_decorator

DataSource = Literal["test", "train", "X_y"]


class _MetricsAccessor(_ComparisonMetricsAccessor):
    """Accessor for metrics-related operations.

    You can access this accessor using the `metrics` attribute.
    """

    def __init__(self, parent: CrossValidationReport) -> None:
        super().__init__(parent)

    ####################################################################################
    # Methods related to displays
    ####################################################################################

    @progress_decorator(description="Computing predictions for display")
    def _get_display(
        self,
        *,
        data_source: DataSource,
        response_method: str,
        display_class: Any,
        display_kwargs: dict[str, Any],
    ) -> Union[RocCurveDisplay, PrecisionRecallCurveDisplay, PredictionErrorDisplay]:
        """Get the display from the cache or compute it.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        response_method : str
            The response method.

        display_class : class
            The display class.

        display_kwargs : dict
            The display kwargs used by `display_class._from_predictions`.

        Returns
        -------
        display : display_class
            The display.
        """
        # Create a list of cache key components and then convert to tuple
        if "random_state" in display_kwargs and display_kwargs["random_state"] is None:
            cache_key = None
        else:
            cache_key_parts: list[Any] = [self._parent._hash, display_class.__name__]
            cache_key_parts.extend(display_kwargs.values())
            cache_key_parts.append(data_source)
            cache_key = tuple(cache_key_parts)

        assert self._progress_info is not None, "Progress info not set"
        progress = self._progress_info["current_progress"]
        main_task = self._progress_info["current_task"]
        total_estimators = len(self._parent.estimator_reports_)
        progress.update(main_task, total=total_estimators)

        if cache_key in self._parent._cache:
            display = self._parent._cache[cache_key]
        else:
            y_true, y_pred = [], []
            for report in self._parent.estimator_reports_:
                X, y, _ = report.metrics._get_X_y_and_data_source_hash(
                    data_source=data_source
                )
                y_true.append(y)
                y_pred.append(
                    _get_cached_response_values(
                        cache=report._cache,
                        estimator_hash=report._hash,
                        estimator=report._estimator,
                        X=X,
                        response_method=response_method,
                        data_source=data_source,
                        data_source_hash=None,
                        pos_label=display_kwargs.get("pos_label"),
                    )
                )
                progress.update(main_task, advance=1, refresh=True)

            display = display_class._from_predictions(
                y_true,
                y_pred,
                estimator=self._parent.estimator_reports_[0]._estimator,
                estimator_name=self._parent.estimator_name_,
                ml_task=self._parent._ml_task,
                data_source=data_source,
                **display_kwargs,
            )

            # Unless random_state is an int (i.e. the call is deterministic),
            # we do not cache
            if cache_key is not None:
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
        data_source: DataSource = "test",
        pos_label: Optional[Union[int, float, bool, str]] = None,
    ) -> RocCurveDisplay:
        """Plot the ROC curve.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        pos_label : int, float, bool or str, default=None
            The positive class.

        Returns
        -------
        RocCurveDisplay
            The ROC curve display.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        >>> display = report.metrics.roc()
        >>> display.plot(roc_curve_kwargs={"color": "tab:red"})
        """
        response_method = ("predict_proba", "decision_function")
        display_kwargs = {"pos_label": pos_label}
        display = self._get_display(
            data_source=data_source,
            response_method=response_method,
            display_class=RocCurveDisplay,
            display_kwargs=display_kwargs,
        )
        assert isinstance(display, RocCurveDisplay)
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
        pos_label: Optional[Union[int, float, bool, str]] = None,
    ) -> PrecisionRecallCurveDisplay:
        """Plot the precision-recall curve.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        pos_label : int, float, bool or str, default=None
            The positive class.

        Returns
        -------
        PrecisionRecallCurveDisplay
            The precision-recall curve display.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import CrossValidationReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = CrossValidationReport(classifier, X=X, y=y, cv_splitter=2)
        >>> display = report.metrics.precision_recall()
        >>> display.plot()
        """
        response_method = ("predict_proba", "decision_function")
        display_kwargs = {"pos_label": pos_label}
        display = self._get_display(
            data_source=data_source,
            response_method=response_method,
            display_class=PrecisionRecallCurveDisplay,
            display_kwargs=display_kwargs,
        )
        assert isinstance(display, PrecisionRecallCurveDisplay)
        return display

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["regression", "multioutput-regression"]
        )
    )
    def prediction_error(
        self,
        *,
        data_source: DataSource = "test",
        subsample: Union[float, int, None] = 1_000,
        random_state: Optional[int] = None,
    ) -> PredictionErrorDisplay:
        """Plot the prediction error of a regression model.

        Extra keyword arguments will be passed to matplotlib's `plot`.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        subsample : float, int or None, default=1_000
            Sampling the samples to be shown on the scatter plot. If `float`,
            it should be between 0 and 1 and represents the proportion of the
            original dataset. If `int`, it represents the number of samples
            display on the scatter plot. If `None`, no subsampling will be
            applied. by default, 1,000 samples or less will be displayed.

        random_state : int, default=None
            The random state to use for the subsampling.

        Returns
        -------
        PredictionErrorDisplay
            The prediction error display.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import CrossValidationReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> report = CrossValidationReport(regressor, X=X, y=y, cv_splitter=2)
        >>> display = report.metrics.prediction_error()
        >>> display.plot(kind="actual_vs_predicted", line_kwargs={"color": "tab:red"})
        """
        display_kwargs = {"subsample": subsample, "random_state": random_state}
        display = self._get_display(
            data_source=data_source,
            response_method="predict",
            display_class=PredictionErrorDisplay,
            display_kwargs=display_kwargs,
        )
        assert isinstance(display, PredictionErrorDisplay)
        return display
