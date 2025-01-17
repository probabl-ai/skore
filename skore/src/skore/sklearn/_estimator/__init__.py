from skore.externals._pandas_accessors import _register_accessor
from skore.sklearn._estimator.metrics_accessor import (
    _MetricsAccessor,
    _PlotMetricsAccessor,
)
from skore.sklearn._estimator.report import EstimatorReport

# add the metrics accessor to the estimator report
_register_accessor("metrics", EstimatorReport)(_MetricsAccessor)

# add the plot accessor to the metrics accessor
_register_accessor("plot", _MetricsAccessor)(_PlotMetricsAccessor)

__all__ = ["EstimatorReport"]
