from skore._externals._pandas_accessors import _register_accessor
from skore._sklearn._cross_validation.data_accessor import _DataAccessor
from skore._sklearn._cross_validation.inspection_accessor import (
    _InspectionAccessor,
)
from skore._sklearn._cross_validation.metrics_accessor import _MetricsAccessor
from skore._sklearn._cross_validation.report import (
    CrossValidationReport,
)

_register_accessor("metrics", CrossValidationReport)(_MetricsAccessor)
_register_accessor("data", CrossValidationReport)(_DataAccessor)
_register_accessor("inspection", CrossValidationReport)(_InspectionAccessor)

__all__ = ["CrossValidationReport"]
