# %%
from skore import EstimatorReport, ComparisonReport
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from skore import train_test_split
X, y = load_breast_cancer(return_X_y=True)
split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
classifier = LogisticRegression()
report_a = EstimatorReport(classifier, pos_label=1, **split_data)
classifier = HistGradientBoostingClassifier()
report_b = EstimatorReport(classifier, pos_label=1, **split_data)
comparison_report = ComparisonReport(
    {"report_a": report_a, "report_b": report_b}
)
display = comparison_report.metrics.summarize()
display.plot(x="roc_auc", y="fit_time")
# %%
