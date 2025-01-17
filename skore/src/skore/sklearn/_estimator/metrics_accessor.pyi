from typing import Any, Callable, Literal, Optional, Union

import matplotlib.axes
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from skore.sklearn._base import _BaseAccessor
from skore.sklearn._plot import PredictionErrorDisplay

class _PlotMetricsAccessor(_BaseAccessor):
    _metrics_parent: _MetricsAccessor

    def __init__(self, parent: _MetricsAccessor) -> None: ...
    def _get_display(
        self,
        *,
        X: Optional[np.ndarray],
        y: Optional[np.ndarray],
        data_source: Literal["test", "train", "X_y"],
        response_method: Union[str, list[str]],
        display_class: Any,
        display_kwargs: dict[str, Any],
        display_plot_kwargs: dict[str, Any],
    ) -> Union[RocCurveDisplay, PrecisionRecallDisplay, PredictionErrorDisplay]: ...
    def roc(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        pos_label: Optional[Union[str, int]] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
    ) -> RocCurveDisplay: ...
    def precision_recall(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        pos_label: Optional[Union[str, int]] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
    ) -> PrecisionRecallDisplay: ...
    def prediction_error(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        kind: Literal[
            "actual_vs_predicted", "residual_vs_predicted"
        ] = "residual_vs_predicted",
        subsample: Optional[Union[int, float]] = 1_000,
    ) -> PredictionErrorDisplay: ...

class _MetricsAccessor(_BaseAccessor):
    _SCORE_OR_LOSS_ICONS: dict[str, str]
    plot: _PlotMetricsAccessor

    def _compute_metric_scores(
        self,
        metric_fn: Callable,
        X: Optional[np.ndarray],
        y_true: Optional[np.ndarray],
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        response_method: Union[str, list[str]],
        pos_label: Optional[Union[str, int]] = None,
        metric_name: Optional[str] = None,
        **metric_kwargs: Any,
    ) -> pd.DataFrame: ...
    def report_metrics(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        scoring: Optional[Union[list[str], Callable]] = None,
        pos_label: Optional[Union[str, int]] = None,
        scoring_kwargs: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame: ...
    def accuracy(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> pd.DataFrame: ...
    def precision(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        average: Optional[
            Literal["binary", "micro", "macro", "weighted", "samples"]
        ] = None,
        pos_label: Optional[Union[str, int]] = None,
    ) -> pd.DataFrame: ...
    def recall(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        average: Optional[
            Literal["binary", "micro", "macro", "weighted", "samples"]
        ] = None,
        pos_label: Optional[Union[str, int]] = None,
    ) -> pd.DataFrame: ...
    def brier_score(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        pos_label: Optional[Union[str, int]] = None,
    ) -> pd.DataFrame: ...
    def roc_auc(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        average: Optional[
            Literal["auto", "micro", "macro", "weighted", "samples"]
        ] = None,
        multi_class: Literal["raise", "ovr", "ovo"] = "ovr",
    ) -> pd.DataFrame: ...
    def log_loss(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> pd.DataFrame: ...
    def r2(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        multioutput: Union[
            Literal["raw_values", "uniform_average"], np.ndarray
        ] = "raw_values",
    ) -> pd.DataFrame: ...
    def rmse(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        multioutput: Union[
            Literal["raw_values", "uniform_average"], np.ndarray
        ] = "raw_values",
    ) -> pd.DataFrame: ...
    def custom_metric(
        self,
        metric_function: Callable,
        response_method: Union[str, list[str]],
        *,
        metric_name: Optional[str] = None,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> pd.DataFrame: ...
