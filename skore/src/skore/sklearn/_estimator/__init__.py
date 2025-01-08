from skore.sklearn._estimator.metrics_accessor import _MetricsAccessor
from skore.sklearn._estimator.report import EstimatorReport
from skore.utils._accessor import register_accessor

register_accessor("metrics", EstimatorReport)(_MetricsAccessor)

__all__ = ["EstimatorReport"]
