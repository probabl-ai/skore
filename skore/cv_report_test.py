# ruff: noqa
import time

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from skore import CrossValidationReport


class SlowEstimator(LogisticRegression):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        time.sleep(0.5)
        super().fit(X, y)
        return self


X, y = make_classification(random_state=42)
estimator = SlowEstimator()

report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=5)

# Terminate it early
report._fit_estimator_reports()

# Should list <5 splits, but still work
report.metrics.report_metrics()
