from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from skore import ComparisonReport, EstimatorReport


def test_coefficients_frame():
    X, y = make_classification(n_features=10, random_state=0)
    estimators = {
        "LinearSVC": LinearSVC(C=2),
        "LogisticRegression": LogisticRegression(),
    }

    est_reports = {
        name: EstimatorReport(est, X_train=X, X_test=X, y_train=y, y_test=y)
        for name, est in estimators.items()
    }
    est_comparison_report = ComparisonReport(est_reports)
    result = est_comparison_report.feature_importance.coefficients().frame()
    assert result.shape == (11, 2)

    expected_index = ["Intercept"] + [f"Feature #{i}" for i in range(X.shape[1])]
    assert result.index.tolist() == expected_index

    expected_columns = list(estimators)
    assert result.columns.tolist() == expected_columns
