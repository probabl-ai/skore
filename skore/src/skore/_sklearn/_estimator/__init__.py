from skore._sklearn._estimator.feature_importance_accessor import (
    _FeatureImportanceAccessor,
)
from skore._sklearn._estimator.metrics_accessor import _MetricsAccessor
from skore._sklearn._estimator.report import EstimatorReport
from skore.externals._pandas_accessors import _register_accessor

_register_accessor("metrics", EstimatorReport)(_MetricsAccessor)

_register_accessor("feature_importance", EstimatorReport)(_FeatureImportanceAccessor)

__all__ = ["EstimatorReport"]
