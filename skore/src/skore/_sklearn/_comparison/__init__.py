from skore._externals._pandas_accessors import _register_accessor
from skore._sklearn._comparison.feature_importance_accessor import (
    _FeatureImportanceAccessor,
)
from skore._sklearn._comparison.metrics_accessor import _MetricsAccessor
from skore._sklearn._comparison.report import ComparisonReport

_register_accessor("metrics", ComparisonReport)(_MetricsAccessor)
_register_accessor("feature_importance", ComparisonReport)(_FeatureImportanceAccessor)

__all__ = ["ComparisonReport"]
