from skore._externals._pandas_accessors import _register_accessor
from skore._sklearn._estimator.data_accessor import _DataAccessor
from skore._sklearn._estimator.feature_importance_accessor import (
    _FeatureImportanceAccessor,
)
from skore._sklearn._estimator.metrics_accessor import _MetricsAccessor
from skore._sklearn._estimator.report import EstimatorReport

_register_accessor("metrics", EstimatorReport)(_MetricsAccessor)

_register_accessor("feature_importance", EstimatorReport)(_FeatureImportanceAccessor)

_register_accessor("data", EstimatorReport)(_DataAccessor)

__all__ = ["EstimatorReport"]
