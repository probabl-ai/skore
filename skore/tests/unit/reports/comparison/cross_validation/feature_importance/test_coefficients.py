from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from skore import ComparisonReport, CrossValidationReport


def test_coefficients_frame():
    X, y = make_classification(n_features=10, random_state=0)
    estimators = {
        "LinearSVC": LinearSVC(C=2),
        "LogisticRegression": LogisticRegression(),
    }

    splitter = 5
    cv_reports = {
        name: CrossValidationReport(est, X=X, y=y, splitter=splitter)
        for name, est in estimators.items()
    }
    est_comparison_report = ComparisonReport(cv_reports)
    result = est_comparison_report.feature_importance.coefficients().frame()
    assert result.shape == (5, 22)

    expected_index = list(range(splitter))
    assert result.index.tolist() == expected_index

    base_columns = ["Intercept"] + [f"Feature #{i}" for i in range(X.shape[1])]
    expected_columns = [
        f"{model}__{col}" for model in estimators for col in base_columns
    ]
    assert result.columns.tolist() == expected_columns
