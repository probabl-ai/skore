from skore._sklearn._comparison.metrics_accessor import _MetricsAccessor
from skore._sklearn._comparison.report import ComparisonReport
from skore.externals._pandas_accessors import _register_accessor

_register_accessor("metrics", ComparisonReport)(_MetricsAccessor)

__all__ = ["ComparisonReport"]
