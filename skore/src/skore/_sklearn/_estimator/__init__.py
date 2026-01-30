from skore._externals._pandas_accessors import _register_accessor
from skore._sklearn._estimator.data_accessor import _DataAccessor
from skore._sklearn._estimator.inspection_accessor import (
    _InspectionAccessor,
)
from skore._sklearn._estimator.metrics_accessor import _MetricsAccessor
from skore._sklearn._estimator.report import EstimatorReport

_register_accessor("metrics", EstimatorReport)(_MetricsAccessor)

_register_accessor("inspection", EstimatorReport)(_InspectionAccessor)

_register_accessor("data", EstimatorReport)(_DataAccessor)

__all__ = ["EstimatorReport"]
