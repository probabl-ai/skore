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
    def f1(
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
    def confusion_matrix(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        normalize: Optional[Literal["true", "pred", "all"]] = None,
    ) -> pd.DataFrame: ...
    def classification_report(
        self,
        *,
        data_source: Literal["test", "train", "X_y"] = "test",
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        digits: int = 2,
    ) -> pd.DataFrame: ...
    def feature_importance(self, *, top_k: Optional[int] = None) -> pd.DataFrame: ...

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
    @classmethod
    def from_fitted_estimator(
        cls, estimator: BaseEstimator, *, X: ndarray, y: Optional[ndarray] = None
    ) -> EstimatorReport: ...
    def _repr_html_(self) -> str: ...
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
