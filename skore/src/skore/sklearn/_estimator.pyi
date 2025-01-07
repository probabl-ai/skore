from typing import Any, Literal, Optional, Union

import matplotlib.axes
import pandas as pd
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from skore.sklearn._plot import PredictionErrorDisplay

class _BaseAccessor:
    _parent: EstimatorReport
    def __init__(self, parent: EstimatorReport, icon: str) -> None: ...
    def help(self) -> None: ...

class _PlotMetricsAccessor(_BaseAccessor):
    def roc(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        pos_label: Optional[Union[str, int]] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        name: Optional[str] = None,
    ) -> RocCurveDisplay: ...
    def precision_recall(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        pos_label: Optional[Union[str, int]] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        name: Optional[str] = None,
    ) -> PrecisionRecallDisplay: ...
    def prediction_error(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        kind: Literal[
            "actual_vs_predicted", "residual_vs_predicted"
        ] = "residual_vs_predicted",
        subsample: Optional[Union[int, float]] = 1_000,
    ) -> PredictionErrorDisplay: ...

class _MetricsAccessor(_BaseAccessor):
    plot: _PlotMetricsAccessor

    def report_metrics(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        scoring: Optional[Union[list[str], callable]] = None,
        pos_label: int = 1,
        scoring_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame: ...
    def custom_metric(
        self,
        metric_function: callable,
        response_method: Union[str, list[str]],
        *,
        metric_name: Optional[str] = None,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        **kwargs: Any,
    ) -> pd.DataFrame: ...
    def accuracy(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
    ) -> pd.DataFrame: ...
    def precision(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        average: Literal[
            "auto", "macro", "micro", "weighted", "samples", None
        ] = "auto",
        pos_label: Optional[Union[str, int]] = None,
    ) -> pd.DataFrame: ...
    def recall(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        average: Literal[
            "auto", "macro", "micro", "weighted", "samples", None
        ] = "auto",
        pos_label: Optional[Union[str, int]] = None,
    ) -> pd.DataFrame: ...
    def brier_score(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        pos_label: int = 1,
    ) -> pd.DataFrame: ...
    def roc_auc(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        average: Literal["auto", "macro", "micro", "weighted", "samples"] = "auto",
        multi_class: Literal["raise", "ovr", "ovo", "auto"] = "ovr",
    ) -> pd.DataFrame: ...
    def log_loss(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
    ) -> pd.DataFrame: ...
    def r2(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        multioutput: Union[
            Literal["raw_values", "uniform_average"], ndarray
        ] = "uniform_average",
    ) -> pd.DataFrame: ...
    def rmse(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        multioutput: Union[
            Literal["raw_values", "uniform_average"], ndarray
        ] = "uniform_average",
    ) -> pd.DataFrame: ...

class EstimatorReport:
    metrics: _MetricsAccessor

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        fit: Literal["auto", True, False] = "auto",
        X_train: Optional[ndarray] = None,
        y_train: Optional[ndarray] = None,
        X_test: Optional[ndarray] = None,
        y_test: Optional[ndarray] = None,
    ) -> None: ...
    def clean_cache(self) -> None: ...
    def cache_predictions(self, response_methods="auto", n_jobs=None) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def estimator(self) -> BaseEstimator: ...
    @property
    def X_train(self) -> Optional[ndarray]: ...
    @property
    def y_train(self) -> Optional[ndarray]: ...
    @property
    def X_test(self) -> Optional[ndarray]: ...
    @property
    def y_test(self) -> Optional[ndarray]: ...
    def help(self) -> None: ...
