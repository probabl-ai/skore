from skore._checks.accessor import _ChecksAccessor
from skore._externals.pandas_accessors import _register_accessor
from skore._reports.comparison.inspection import (
    _InspectionAccessor,
)
from skore._reports.comparison.metrics import _MetricsAccessor
from skore._reports.comparison.report import ComparisonReport

_register_accessor("metrics", ComparisonReport)(_MetricsAccessor)
_register_accessor("inspection", ComparisonReport)(_InspectionAccessor)
_register_accessor("checks", ComparisonReport)(_ChecksAccessor)

__all__ = ["ComparisonReport"]
