from skore.externals._pandas_accessors import _register_accessor

from .metrics_accessor import _MetricsAccessor
from .report import CrossValidationComparisonReport

_register_accessor("metrics", CrossValidationComparisonReport)(_MetricsAccessor)

__all__ = ["CrossValidationComparisonReport"]
