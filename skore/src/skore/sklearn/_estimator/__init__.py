from skore.externals._pandas_accessors import _register_accessor
from skore.sklearn._estimator.metrics_accessor import _MetricsAccessor
from skore.sklearn._estimator.report import EstimatorReport

_register_accessor("metrics", EstimatorReport)(_MetricsAccessor)

__all__ = ["EstimatorReport"]
