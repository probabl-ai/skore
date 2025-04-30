from skore.externals._pandas_accessors import _register_accessor
from skore.sklearn._estimator.feature_importance_accessor import (
    _FeatureImportanceAccessor,
)
from skore.sklearn._estimator.metrics_accessor import _MetricsAccessor
from skore.sklearn._estimator.report import EstimatorReport

_register_accessor("metrics", EstimatorReport)(_MetricsAccessor)

_register_accessor("feature_importance", EstimatorReport)(_FeatureImportanceAccessor)

__all__ = ["EstimatorReport"]
