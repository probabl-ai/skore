from skore._checks.accessor import _ChecksAccessor
from skore._externals.pandas_accessors import _register_accessor
from skore._reports.cross_validation.data import _DataAccessor
from skore._reports.cross_validation.inspection import (
    _InspectionAccessor,
)
from skore._reports.cross_validation.metrics import _MetricsAccessor
from skore._reports.cross_validation.report import (
    CrossValidationReport,
)

_register_accessor("metrics", CrossValidationReport)(_MetricsAccessor)
_register_accessor("data", CrossValidationReport)(_DataAccessor)
_register_accessor("inspection", CrossValidationReport)(_InspectionAccessor)
_register_accessor("checks", CrossValidationReport)(_ChecksAccessor)

__all__ = ["CrossValidationReport"]
