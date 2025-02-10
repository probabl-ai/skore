from skore.externals._pandas_accessors import _register_accessor
from skore.sklearn._comparison.metrics_accessor import _MetricsAccessor
from skore.sklearn._comparison.report import ComparisonReport

_register_accessor("metrics", ComparisonReport)(_MetricsAccessor)

__all__ = ["ComparisonReport"]
