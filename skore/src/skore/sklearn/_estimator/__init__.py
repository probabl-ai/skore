from skore.sklearn._estimator.metrics_accessor import _MetricsAccessor
from skore.sklearn._estimator.report import EstimatorReport
from skore.utils._accessor import _register_accessor, doc


@doc(_register_accessor, klass="EstimatorReport")
def register_estimator_report_accessor(name: str):
    return _register_accessor(name, EstimatorReport)


register_estimator_report_accessor("metrics")(_MetricsAccessor)

__all__ = ["EstimatorReport"]
