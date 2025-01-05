import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore.sklearn._plot import RocCurveDisplay


@pytest.fixture
def binary_classification_data():
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return LogisticRegression().fit(X_train, y_train), X_train, X_test, y_train, y_test


@pytest.fixture
def multiclass_classification_data():
    X, y = make_classification(n_classes=3, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return LogisticRegression().fit(X_train, y_train), X_train, X_test, y_train, y_test


def test_roc_curve_display_binary_classification(pyplot, binary_classification_data):
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.plot.roc()
    assert isinstance(display, RocCurveDisplay)

    # check the structure of the attributes
    for attr_name in ("fpr", "tpr", "roc_auc"):
        assert isinstance(getattr(display, attr_name), dict)
        assert len(getattr(display, attr_name)) == 1

        attr = getattr(display, attr_name)
        assert list(attr.keys()) == [estimator.classes_[1]]
        assert list(attr.keys()) == [display.pos_label]
        assert isinstance(attr[estimator.classes_[1]], list)
        assert len(attr[estimator.classes_[1]]) == 1

    # check the default plotting behaviour
    import matplotlib as mpl

    assert isinstance(display.lines_, list)
    assert len(display.lines_) == 1
    roc_curve_mpl = display.lines_[0]
    assert isinstance(roc_curve_mpl, mpl.lines.Line2D)
    assert (
        roc_curve_mpl.get_label()
        == f"Test set (AUC = {display.roc_auc[estimator.classes_[1]][0]:0.2f})"
    )
    assert roc_curve_mpl.get_color() == "#1f77b4"  # tab:blue in hex

    assert isinstance(display.chance_level_, mpl.lines.Line2D)
    assert display.chance_level_.get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_.get_color() == "k"

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == estimator.__class__.__name__
    assert len(legend.get_texts()) == 2

    assert display.ax_.get_xlabel() == "False Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "True Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_roc_curve_display_multiclass_classification(
    pyplot, multiclass_classification_data
):
    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.plot.roc()
    assert isinstance(display, RocCurveDisplay)

    # check the structure of the attributes
    for attr_name in ("fpr", "tpr", "roc_auc"):
        assert isinstance(getattr(display, attr_name), dict)
        assert len(getattr(display, attr_name)) == len(estimator.classes_)

        attr = getattr(display, attr_name)
        for class_label in estimator.classes_:
            assert isinstance(attr[class_label], list)
            assert len(attr[class_label]) == 1

    # check the default plotting behaviour
    import matplotlib as mpl

    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(estimator.classes_)
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for class_label, expected_color in zip(estimator.classes_, default_colors):
        roc_curve_mpl = display.lines_[class_label]
        assert isinstance(roc_curve_mpl, mpl.lines.Line2D)
        assert roc_curve_mpl.get_label() == (
            f"{str(class_label).title()} test set "
            f"(AUC = {display.roc_auc[class_label][0]:0.2f})"
        )
        assert roc_curve_mpl.get_color() == expected_color

    assert isinstance(display.chance_level_, mpl.lines.Line2D)
    assert display.chance_level_.get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_.get_color() == "k"

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == estimator.__class__.__name__
    assert len(legend.get_texts()) == 4

    assert display.ax_.get_xlabel() == "False Positive Rate"
    assert display.ax_.get_ylabel() == "True Positive Rate"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_roc_curve_display_plot_error_wrong_roc_curve_kwargs(
    pyplot, binary_classification_data, multiclass_classification_data
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `roc_curve_kwargs` argument.
    """
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.plot.roc()
    err_msg = (
        "You intend to plot a single ROC curve and provide multiple ROC curve "
        "keyword arguments"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(roc_curve_kwargs=[{}, {}])

    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.plot.roc()
    err_msg = "You intend to plot multiple ROC curves."
    with pytest.raises(ValueError, match=err_msg):
        display.plot(roc_curve_kwargs=[{}, {}])

    with pytest.raises(ValueError, match=err_msg):
        display.plot(roc_curve_kwargs={})
