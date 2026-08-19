from skore._externals._pandas_accessors import _register_accessor
from skore.checks.accessor import _ChecksAccessor
from skore.reports.cross_validation.data import _DataAccessor
from skore.reports.cross_validation.inspection import (
    _InspectionAccessor,
)
from skore.reports.cross_validation.metrics import _MetricsAccessor
from skore.reports.cross_validation.report import (
    CrossValidationReport,
)

_register_accessor("metrics", CrossValidationReport)(_MetricsAccessor)
_register_accessor("data", CrossValidationReport)(_DataAccessor)
_register_accessor("inspection", CrossValidationReport)(_InspectionAccessor)
_register_accessor("checks", CrossValidationReport)(_ChecksAccessor)

__all__ = ["CrossValidationReport"]
