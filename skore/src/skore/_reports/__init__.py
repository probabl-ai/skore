"""Report classes for evaluating and comparing estimators."""

from typing import TYPE_CHECKING

from skore._externals import lazy_loader

if TYPE_CHECKING:
    from skore._reports.comparison import ComparisonReport
    from skore._reports.cross_validation import CrossValidationReport
    from skore._reports.estimator import EstimatorReport

__all__ = ["ComparisonReport", "CrossValidationReport", "EstimatorReport"]

__getattr__, __dir__, _ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "comparison": ["ComparisonReport"],
        "cross_validation": ["CrossValidationReport"],
        "estimator": ["EstimatorReport"],
    },
)
