from skore._externals._pandas_accessors import _register_accessor
from skore.checks.accessor import _ChecksAccessor
from skore.reports.comparison.inspection import (
    _InspectionAccessor,
)
from skore.reports.comparison.metrics import _MetricsAccessor
from skore.reports.comparison.report import ComparisonReport

_register_accessor("metrics", ComparisonReport)(_MetricsAccessor)
_register_accessor("inspection", ComparisonReport)(_InspectionAccessor)
_register_accessor("checks", ComparisonReport)(_ChecksAccessor)

__all__ = ["ComparisonReport"]
