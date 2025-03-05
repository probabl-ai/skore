from typing import cast

from skore.externals._pandas_accessors import Accessor
from skore.sklearn._cross_validation.metrics_accessor import _MetricsAccessor
from skore.sklearn._cross_validation.report import (
    CrossValidationReport,
)

CrossValidationReport.metrics = cast(
    _MetricsAccessor, Accessor("metrics", _MetricsAccessor)
)

__all__ = ["CrossValidationReport"]
