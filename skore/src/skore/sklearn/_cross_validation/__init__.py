from skore.externals._pandas_accessors import _register_accessor
from skore.sklearn._cross_validation.metrics_accessor import (
    _MetricsAccessor,
    _PlotMetricsAccessor,
)
from skore.sklearn._cross_validation.report import (
    CrossValidationReport,
)

# add the metrics accessor to the estimator report
_register_accessor("metrics", CrossValidationReport)(_MetricsAccessor)

# add the plot accessor to the metrics accessor
_register_accessor("plot", _MetricsAccessor)(_PlotMetricsAccessor)

__all__ = ["CrossValidationReport"]
