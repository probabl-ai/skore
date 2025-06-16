import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from skore import CrossValidationReport, EstimatorReport


@pytest.fixture
def estimator_report_classification():
    X, y = make_classification(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    estimator_report = EstimatorReport(
        estimator=HistGradientBoostingClassifier(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    return estimator_report


def test_not_implemented_estimator(estimator_report_classification):
    """
    Test that the plot_comparison_estimator method raises NotImplementedError
    when called with a binary classification comparator.
    """
    with pytest.raises(NotImplementedError):
        estimator_report_classification.metrics.summarize().plot(
            x="accuracy", y="f1_score"
        )


def test_not_implemented_other_categories():
    """
    Test that the plot_comparison_estimator method raises NotImplementedError
    when called with a binary classification comparator.
    """
    X, y = make_classification(random_state=0)
    cv_report = CrossValidationReport(
        estimator=HistGradientBoostingClassifier(),
        X=X,
        y=y,
    )
    with pytest.raises(NotImplementedError, match="To come soon!"):
        cv_report.metrics.summarize().plot(x="accuracy", y="f1_score")
