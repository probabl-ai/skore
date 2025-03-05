from typing import cast

from skore.externals._pandas_accessors import Accessor
from skore.sklearn._estimator.feature_importance_accessor import (
    _FeatureImportanceAccessor,
)
from skore.sklearn._estimator.metrics_accessor import _MetricsAccessor
from skore.sklearn._estimator.report import EstimatorReport

EstimatorReport.metrics = cast(_MetricsAccessor, Accessor("metrics", _MetricsAccessor))

EstimatorReport.feature_importance = cast(
    _FeatureImportanceAccessor,
    Accessor("feature_importance", _FeatureImportanceAccessor),
)

__all__ = ["EstimatorReport"]
