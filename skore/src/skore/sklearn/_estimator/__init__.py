from skore.externals._pandas_accessors import _register_accessor
from skore.sklearn._estimator.metrics_accessor import (
    _MetricsAccessor,
    _PlotMetricsAccessor,
)
from skore.sklearn._estimator.report import EstimatorReport


def register_estimator_report_accessor(name: str):
    """Register an accessor for the EstimatorReport class."""
    return _register_accessor(name, EstimatorReport)


def register_estimator_report_metrics_accessor(name: str):
    """Register an accessor for the EstimatorReport class."""
    return _register_accessor(name, _MetricsAccessor)


# add the plot accessor to the metrics accessor
register_estimator_report_metrics_accessor("plot")(_PlotMetricsAccessor)

# add the metrics accessor to the estimator report
register_estimator_report_accessor("metrics")(_MetricsAccessor)

__all__ = ["EstimatorReport"]
