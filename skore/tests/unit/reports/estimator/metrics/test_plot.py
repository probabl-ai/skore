import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from skore import EstimatorReport, RocCurveDisplay


def test_plot_roc(forest_binary_classification_with_test):
    """Check that the ROC plot method works."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert isinstance(report.metrics.roc(), RocCurveDisplay)


@pytest.mark.parametrize("display", ["roc", "precision_recall"])
def test_display_binary_classification(
    pyplot, forest_binary_classification_with_test, display
):
    """The call to display functions should be cached."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)()
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)()
    assert display_first_call is display_second_call


@pytest.mark.parametrize("metric", ["roc", "precision_recall"])
def test_display_binary_classification_pos_label(pyplot, metric):
    """Check the behaviour of the display methods when `pos_label` needs to be set."""
    X, y = make_classification(
        n_classes=2, class_sep=0.8, weights=[0.4, 0.6], random_state=0
    )
    labels = np.array(["A", "B"], dtype=object)
    y = labels[y]
    classifier = LogisticRegression().fit(X, y)
    report = EstimatorReport(classifier, X_test=X, y_test=y)
    display = getattr(report.metrics, metric)()
    fig = display.plot()
    assert "Positive label" not in fig.get_suptitle()

    report = EstimatorReport(classifier, X_test=X, y_test=y, pos_label="A")
    display = getattr(report.metrics, metric)()
    fig = display.plot()
    assert "Positive label: A" in fig.get_suptitle()


@pytest.mark.parametrize("metric", ["roc", "precision_recall"])
def test_display_binary_classification_decision_function_default_pos_label(
    pyplot, metric
):
    """Check that the default binary behaviour works with 1D decision scores."""
    X, y = make_classification(
        n_classes=2,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=0,
    )
    classifier = LinearSVC(dual=False, random_state=0).fit(X, y)
    report = EstimatorReport(classifier, X_test=X, y_test=y)

    display = getattr(report.metrics, metric)()
    frame = (
        display.frame(with_roc_auc=True)
        if metric == "roc"
        else display.frame(with_average_precision=True)
    )

    assert "label" in frame.columns
    np.testing.assert_array_equal(frame["label"].cat.categories, classifier.classes_)

    fig = display.plot()
    legend = fig.axes[0].get_legend()
    assert legend is not None
    expected_n_entries = len(classifier.classes_) + (1 if metric == "roc" else 0)
    assert len(legend.get_texts()) == expected_n_entries


@pytest.mark.parametrize("display", ["prediction_error"])
def test_display_regression(pyplot, linear_regression_with_test, display):
    """The call to display functions should be cached, as long as the arguments make it
    reproducible."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)(seed=0)
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)(seed=0)
    assert display_first_call is display_second_call


@pytest.mark.parametrize("display", ["roc", "precision_recall"])
def test_display_binary_classification_switching_data_source(
    pyplot, forest_binary_classification_with_test, display
):
    """Check that we don't hit the cache when switching the data source."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(
        estimator, X_train=X_test, y_train=y_test, X_test=X_test, y_test=y_test
    )
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)(data_source="test")
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)(data_source="train")
    assert display_first_call is not display_second_call


@pytest.mark.parametrize("display", ["prediction_error"])
def test_display_regression_switching_data_source(
    pyplot, linear_regression_with_test, display
):
    """Check that we don't hit the cache when switching the data source."""
    estimator, X_test, y_test = linear_regression_with_test
    report = EstimatorReport(
        estimator, X_train=X_test, y_train=y_test, X_test=X_test, y_test=y_test
    )
    assert hasattr(report.metrics, display)
    display_first_call = getattr(report.metrics, display)(data_source="test")
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)(data_source="train")
    assert display_first_call is not display_second_call


def test_display_confusion_matrix_data_source_both_is_not_supported(
    pyplot, forest_binary_classification_with_test
):
    """Check that confusion_matrix rejects data_source='both' explicitly."""
    estimator, X_test, y_test = forest_binary_classification_with_test
    report = EstimatorReport(
        estimator, X_train=X_test, y_train=y_test, X_test=X_test, y_test=y_test
    )

    with pytest.raises(
        ValueError,
        match="data_source='both' is not supported for confusion_matrix.",
    ):
        report.metrics.confusion_matrix(data_source="both")
