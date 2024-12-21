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
from skore.sklearn._plot import RocCurveDisplay


@pytest.fixture
def binary_classification_data():
    """Create a binary classification dataset and return fitted estimator and data."""
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return RandomForestClassifier().fit(X_train, y_train), X_test, y_test


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


###############################################################################
# Check the general behaviour of the report
###############################################################################


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

    err_msg = "attribute is immutable"
    with pytest.raises(AttributeError, match=err_msg):
        report.estimator = LinearRegression()
    with pytest.raises(AttributeError, match=err_msg):
        report.X_train = X_train
    with pytest.raises(AttributeError, match=err_msg):
        report.y_train = y_train


def test_estimator_report_from_fitted_estimator(binary_classification_data):
    """Check the general behaviour of `from_fitted_estimator`."""
    estimator, X, y = binary_classification_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)

    assert report.estimator is estimator  # we should not clone the estimator
    assert report.X_train is None
    assert report.y_train is None
    assert report.X_val is X
    assert report.y_val is y

    err_msg = "attribute is immutable"
    with pytest.raises(AttributeError, match=err_msg):
        report.estimator = LinearRegression()
    with pytest.raises(AttributeError, match=err_msg):
        report.X_train = X
    with pytest.raises(AttributeError, match=err_msg):
        report.y_train = y


def test_estimator_report_invalidate_cache_data(binary_classification_data):
    """Check that we invalidate the cache when the data is changed."""
    estimator, X, y = binary_classification_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)

    for attribute in ("X_val", "y_val"):
        current_hash = report._hash
        report._cache["mocking"] = "mocking"  # mock writing to cache
        setattr(report, attribute, None)
        assert report._cache == {}
        assert report._hash != current_hash


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


def test_estimator_report_help(capsys, binary_classification_data):
    """Check that the help method writes to the console."""
    estimator, X, y = binary_classification_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)

    report.help()
    captured = capsys.readouterr()
    assert (
        f"🔧 Available tools for diagnosing {estimator.__class__.__name__} estimator"
        in captured.out
    )


def test_estimator_report_repr(binary_classification_data):
    """Check that __repr__ returns a string starting with the expected prefix."""
    estimator, X, y = binary_classification_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)

    repr_str = repr(report)
    assert repr_str.startswith("📓 Estimator Reporter")


###############################################################################
# Check the plot methods
###############################################################################


def test_estimator_report_plot_roc(binary_classification_data):
    """Check that the ROC plot method works."""
    estimator, X, y = binary_classification_data
    report = EstimatorReport.from_fitted_estimator(estimator, X=X, y=y)
    assert isinstance(report.plot.roc(), RocCurveDisplay)
