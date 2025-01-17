from skore.externals._pandas_accessors import _register_accessor
from skore.sklearn._cross_validation.metrics_accessor import (
    _MetricsAccessor,
    _PlotMetricsAccessor,
)
from skore.sklearn._cross_validation.report import (
    CrossValidationReport,
)


# add the plot accessor to the metrics accessor
_register_accessor("plot", CrossValidationReport)(_PlotMetricsAccessor)

# add the metrics accessor to the estimator report
_register_accessor("metrics", CrossValidationReport)(_MetricsAccessor)


__all__ = ["CrossValidationReport"]
