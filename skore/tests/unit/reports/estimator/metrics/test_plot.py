import numpy as np
import pytest
from pandas.testing import assert_frame_equal
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
    """Check the behaviour of the display methods when `pos_label` is not set."""
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
def test_display_binary_classification_decision_function_report_pos_label(
    pyplot, metric, binary_classification_data
):
    """Check that the default binary behaviour works with 1D decision scores."""
    X, y = binary_classification_data
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


@pytest.mark.parametrize("metric", ["roc", "precision_recall"])
def test_display_binary_classification_label_overrides_report_pos_label(
    metric, binary_classification_data
):
    """Check that `label` overrides the report positive label for plots and frames."""
    X, y = binary_classification_data
    labels = np.array(["A", "B"])
    y = labels[y]
    classifier = LogisticRegression().fit(X, y)
    report = EstimatorReport(classifier, X_test=X, y_test=y, pos_label="A")
    display = getattr(report.metrics, metric)()

    frame_kwargs = (
        {"with_roc_auc": True} if metric == "roc" else {"with_average_precision": True}
    )
    all_curves = display.frame(label=None, **frame_kwargs)
    selected = display.frame(label="B", **frame_kwargs)
    expected = all_curves.query("label == 'B'").drop(columns="label")
    expected = expected.reset_index(drop=True)

    assert_frame_equal(
        display.frame(**frame_kwargs), display.frame(label="A", **frame_kwargs)
    )
    assert_frame_equal(selected, expected)
    assert "Positive label: B" in display.plot(label="B").get_suptitle()


@pytest.mark.parametrize("metric", ["roc", "precision_recall"])
def test_display_multiclass_label_selects_curve_and_validates(
    metric, multiclass_classification_data
):
    """Check multiclass label selection and validation for plots and frames."""
    X, y = multiclass_classification_data
    classifier = LogisticRegression().fit(X, y)
    report = EstimatorReport(classifier, X_test=X, y_test=y)
    display = getattr(report.metrics, metric)()

    frame_kwargs = (
        {"with_roc_auc": True} if metric == "roc" else {"with_average_precision": True}
    )
    all_curves = display.frame(label=None, **frame_kwargs)
    selected = display.frame(label=1, **frame_kwargs)
    expected = all_curves.query("label == 1").drop(columns="label")
    expected = expected.reset_index(drop=True)
    assert_frame_equal(selected, expected)

    assert "Label: 1" in display.plot(label=1).get_suptitle()
    # check label normalization:
    assert "Label: 1" in display.plot(label=True).get_suptitle()

    err_msg = "label='invalid' is not a valid label"
    with pytest.raises(ValueError, match=err_msg):
        display.frame(label="invalid")
    with pytest.raises(ValueError, match=err_msg):
        display.plot(label="invalid")


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
    display_first_call = getattr(report.metrics, display)(data_source="test", seed=0)
    assert report._cache != {}
    display_second_call = getattr(report.metrics, display)(data_source="train", seed=0)
    assert display_first_call is not display_second_call
