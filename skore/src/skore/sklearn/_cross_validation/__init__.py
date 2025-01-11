from skore.externals._pandas_accessors import _register_accessor
from skore.sklearn._cross_validation.metrics_accessor import (
    _MetricsAccessor,
    _PlotMetricsAccessor,
)
from skore.sklearn._cross_validation.report import (
    CrossValidationReport,
)


def register_cross_validation_report_accessor(name):
    """Register an accessor for the EstimatorReport class."""
    return _register_accessor(name, CrossValidationReport)


def register_cross_validation_report_metrics_accessor(name):
    """Register an accessor for the EstimatorReport class."""
    return _register_accessor(name, _MetricsAccessor)


# add the plot accessor to the metrics accessor
register_cross_validation_report_metrics_accessor("plot")(_PlotMetricsAccessor)

# add the metrics accessor to the estimator report
register_cross_validation_report_accessor("metrics")(_MetricsAccessor)


__all__ = ["CrossValidationReport"]
