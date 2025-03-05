from typing import cast

from skore.externals._pandas_accessors import Accessor
from skore.sklearn._comparison.metrics_accessor import _MetricsAccessor
from skore.sklearn._comparison.report import ComparisonReport

ComparisonReport.metrics = cast(_MetricsAccessor, Accessor("metrics", _MetricsAccessor))

__all__ = ["ComparisonReport"]
