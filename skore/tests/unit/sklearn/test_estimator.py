import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from skore import EstimatorReport
from skore.sklearn._estimator import _check_supported_estimator


def test_check_supported_estimator():
    """Test the behaviour of `_check_supported_estimator`."""

    class MockParent:
        def __init__(self, estimator):
            self.estimator = estimator

    class MockAccessor:
        def __init__(self, parent):
            self._parent = parent

    parent = MockParent(LogisticRegression())
    accessor = MockAccessor(parent)
    check = _check_supported_estimator((LogisticRegression,))
    assert check(accessor)

    pipeline = Pipeline([("clf", LogisticRegression())])
    parent = MockParent(pipeline)
    accessor = MockAccessor(parent)
    assert check(accessor)

    parent = MockParent(RandomForestClassifier())
    accessor = MockAccessor(parent)
    err_msg = (
        "The RandomForestClassifier estimator is not supported by the function called."
    )
    with pytest.raises(AttributeError, match=err_msg):
        check(accessor)


def test_estimator_not_fitted():
    """Test that an error is raised when trying to create a report from an unfitted
    estimator.
    """
    estimator = LinearRegression()
    with pytest.raises(NotFittedError):
        EstimatorReport.from_fitted_estimator(estimator, X=None, y=None)


def test_estimator_report_from_unfitted_estimator():
    """Check the general behaviour of `from_unfitted_estimator`."""
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    estimator = LinearRegression()
    report = EstimatorReport.from_unfitted_estimator(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    check_is_fitted(report.estimator)
    assert report.estimator is not estimator  # the estimator should be cloned

    assert report.X_train is X_train
    assert report.y_train is y_train
    assert report.X_val is X_test
    assert report.y_val is y_test


def test_estimator_report_from_fitted_estimator():
    """Check the general behaviour of `from_fitted_estimator`."""
    X, y = make_regression(random_state=42)
    estimator = LinearRegression().fit(X, y)
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)

    assert report.estimator is estimator  # we should not clone the estimator
    assert report.X_train is None
    assert report.y_train is None
    assert report.X_val is X
    assert report.y_val is y


@pytest.mark.parametrize(
    "X, y, supported_plot_methods, not_supported_plot_methods",
    [
        (*make_classification(random_state=42), ["roc"], []),
        (
            *make_classification(n_classes=3, n_clusters_per_class=1, random_state=42),
            [],
            ["roc"],
        ),
    ],
)
def test_estimator_report_check_support_plot(
    X, y, supported_plot_methods, not_supported_plot_methods
):
    """Check that the available plot methods are correctly registered."""
    classifier = RandomForestClassifier().fit(X, y)
    report = EstimatorReport.from_fitted_estimator(classifier, X=X, y=y)

    for supported_plot_method in supported_plot_methods:
        assert hasattr(report.plot, supported_plot_method)

    for not_supported_plot_method in not_supported_plot_methods:
        assert not hasattr(report.plot, not_supported_plot_method)
