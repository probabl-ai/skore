from skore.sklearn._estimator.metrics_accessor import _MetricsAccessor
from skore.sklearn._estimator.report import EstimatorReport
from skore.utils._accessor import _register_accessor


def register_estimator_report_accessor(name: str):
    """Register an accessor for the EstimatorReport class."""
    return _register_accessor(name, EstimatorReport)


register_estimator_report_accessor("metrics")(_MetricsAccessor)

__all__ = ["EstimatorReport"]
