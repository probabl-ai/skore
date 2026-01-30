from skore._externals._pandas_accessors import _register_accessor
from skore._sklearn._comparison.inspection_accessor import (
    _InspectionAccessor,
)
from skore._sklearn._comparison.metrics_accessor import _MetricsAccessor
from skore._sklearn._comparison.report import ComparisonReport

_register_accessor("metrics", ComparisonReport)(_MetricsAccessor)
_register_accessor("inspection", ComparisonReport)(_InspectionAccessor)

__all__ = ["ComparisonReport"]
