from typing import Literal, Optional, Union

import matplotlib.axes
import pandas as pd
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

class _BaseAccessor:
    _report: EstimatorReport
    def __init__(self, report: EstimatorReport) -> None: ...
    def help(self) -> None: ...

class _PlotAccessor(_BaseAccessor):
    def roc(
        self,
        *,
        pos_label: Optional[Union[str, int]] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        name: Optional[str] = None,
    ) -> RocCurveDisplay: ...
    def precision_recall(
        self,
        *,
        pos_label: Optional[Union[str, int]] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        name: Optional[str] = None,
    ) -> PrecisionRecallDisplay: ...
    def confusion_matrix(
        self,
        *,
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        normalize: Optional[Literal["true", "pred", "all"]] = None,
        display_labels: Optional[list[str]] = None,
        include_values: bool = True,
        values_format: Optional[str] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        colorbar: bool = True,
    ) -> matplotlib.axes.Axes: ...
    def feature_importance(
        self, *, top_k: Optional[int] = None, ax: Optional[matplotlib.axes.Axes] = None
    ) -> matplotlib.axes.Axes: ...

class _MetricsAccessor(_BaseAccessor):
    def accuracy(
        self, *, X: Optional[ndarray] = None, y: Optional[ndarray] = None
    ) -> pd.DataFrame: ...
    def precision(
        self,
        *,
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
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        normalize: Optional[Literal["true", "pred", "all"]] = None,
    ) -> pd.DataFrame: ...
    def classification_report(
        self,
        *,
        X: Optional[ndarray] = None,
        y: Optional[ndarray] = None,
        digits: int = 2,
    ) -> pd.DataFrame: ...
    def feature_importance(self, *, top_k: Optional[int] = None) -> pd.DataFrame: ...

class EstimatorReport:
    metrics: _MetricsAccessor
    plot: _PlotAccessor

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        X_train: Optional[ndarray] = None,
        y_train: Optional[ndarray] = None,
        X_val: Optional[ndarray] = None,
        y_val: Optional[ndarray] = None,
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
    def X_val(self) -> Optional[ndarray]: ...
    @property
    def y_val(self) -> Optional[ndarray]: ...
    def help(self) -> None: ...
