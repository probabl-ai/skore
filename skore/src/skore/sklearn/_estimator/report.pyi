from typing import Any, Literal, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator

from skore.sklearn._base import _HelpMixin
from skore.sklearn._estimator.metrics_accessor import _MetricsAccessor

class EstimatorReport(_HelpMixin):
    _ACCESSOR_CONFIG: dict[str, dict[str, str]]
    _estimator: BaseEstimator
    _X_train: Optional[np.ndarray]
    _y_train: Optional[np.ndarray]
    _X_test: Optional[np.ndarray]
    _y_test: Optional[np.ndarray]
    _rng: np.random.Generator
    _hash: int
    _cache: dict[Any, Any]
    _ml_task: str
    metrics: _MetricsAccessor

    @staticmethod
    def _fit_estimator(
        estimator: BaseEstimator, X_train: np.ndarray, y_train: Optional[np.ndarray]
    ) -> BaseEstimator: ...
    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        fit: Literal["auto", True, False] = "auto",
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> None: ...
    def _initialize_state(self) -> None: ...
    def clear_cache(self) -> None: ...
    def cache_predictions(
        self,
        response_methods: Union[Literal["auto"], list[str]] = "auto",
        n_jobs: Optional[int] = None,
    ) -> None: ...
    @property
    def estimator(self) -> BaseEstimator: ...
    @property
    def X_train(self) -> Optional[np.ndarray]: ...
    @property
    def y_train(self) -> Optional[np.ndarray]: ...
    @property
    def X_test(self) -> Optional[np.ndarray]: ...
    @X_test.setter
    def X_test(self, value: Optional[np.ndarray]) -> None: ...
    @property
    def y_test(self) -> Optional[np.ndarray]: ...
    @y_test.setter
    def y_test(self, value: Optional[np.ndarray]) -> None: ...
    @property
    def estimator_name(self) -> str: ...
    def _get_help_panel_title(self) -> str: ...
    def _create_help_tree(self) -> Any: ...  # Returns rich.tree.Tree
