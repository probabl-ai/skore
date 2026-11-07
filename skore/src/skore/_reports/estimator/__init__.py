from skore._checks.accessor import _ChecksAccessor
from skore._externals.pandas_accessors import _register_accessor
from skore._reports.estimator.data import _DataAccessor
from skore._reports.estimator.inspection import (
    _InspectionAccessor,
)
from skore._reports.estimator.metrics import _MetricsAccessor
from skore._reports.estimator.report import EstimatorReport

_register_accessor("metrics", EstimatorReport)(_MetricsAccessor)
_register_accessor("inspection", EstimatorReport)(_InspectionAccessor)
_register_accessor("data", EstimatorReport)(_DataAccessor)
_register_accessor("checks", EstimatorReport)(_ChecksAccessor)

__all__ = ["EstimatorReport"]
