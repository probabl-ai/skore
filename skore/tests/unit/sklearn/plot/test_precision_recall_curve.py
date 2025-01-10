import matplotlib as mpl
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore.sklearn._plot import PrecisionRecallCurveDisplay


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


def test_precision_recall_curve_display_binary_classification(
    pyplot, binary_classification_data
):
    """Check the attributes and default plotting behaviour of the
    precision-recall curve plot with binary data.
    """
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.plot.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)

    # check the structure of the attributes
    for attr_name in ("precision", "recall", "average_precision", "prevalence"):
        assert isinstance(getattr(display, attr_name), dict)
        assert len(getattr(display, attr_name)) == 1

        attr = getattr(display, attr_name)
        assert list(attr.keys()) == [estimator.classes_[1]]
        assert list(attr.keys()) == [display.pos_label]
        assert isinstance(attr[estimator.classes_[1]], list)
        assert len(attr[estimator.classes_[1]]) == 1

    assert isinstance(display.lines_, list)
    assert len(display.lines_) == 1
    precision_recall_curve_mpl = display.lines_[0]
    assert isinstance(precision_recall_curve_mpl, mpl.lines.Line2D)
    assert (
        precision_recall_curve_mpl.get_label()
        == f"Test set (AP = {display.average_precision[estimator.classes_[1]][0]:0.2f})"
    )
    assert precision_recall_curve_mpl.get_color() == "#1f77b4"  # tab:blue in hex

    assert display.chance_levels_ is None

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == estimator.__class__.__name__
    assert len(legend.get_texts()) == 1

    assert display.ax_.get_xlabel() == "Recall\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "Precision\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_precision_recall_curve_display_data_source(pyplot, binary_classification_data):
    """Check that we can pass the `data_source` argument to the precision-recall
    curve plot.
    """
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.plot.precision_recall(data_source="train")
    assert display.lines_[0].get_label() == "Train set (AP = 1.00)"

    display = report.metrics.plot.precision_recall(
        data_source="X_y", X=X_train, y=y_train
    )
    assert display.lines_[0].get_label() == "AP = 1.00"


def test_precision_recall_curve_display_multiclass_classification(
    pyplot, multiclass_classification_data
):
    """Check the attributes and default plotting behaviour of the precision-recall
    curve plot with multiclass data.
    """
    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.plot.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)

    # check the structure of the attributes
    for attr_name in ("precision", "recall", "average_precision", "prevalence"):
        assert isinstance(getattr(display, attr_name), dict)
        assert len(getattr(display, attr_name)) == len(estimator.classes_)

        attr = getattr(display, attr_name)
        for class_label in estimator.classes_:
            assert isinstance(attr[class_label], list)
            assert len(attr[class_label]) == 1

    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(estimator.classes_)
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for class_label, expected_color in zip(estimator.classes_, default_colors):
        precision_recall_curve_mpl = display.lines_[class_label]
        assert isinstance(precision_recall_curve_mpl, mpl.lines.Line2D)
        assert precision_recall_curve_mpl.get_label() == (
            f"{str(class_label).title()} - test set "
            f"(AP = {display.average_precision[class_label][0]:0.2f})"
        )
        assert precision_recall_curve_mpl.get_color() == expected_color

    assert display.chance_levels_ is None

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == estimator.__class__.__name__
    assert len(legend.get_texts()) == 3

    assert display.ax_.get_xlabel() == "Recall"
    assert display.ax_.get_ylabel() == "Precision"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_precision_recall_curve_display_pr_curve_kwargs(
    pyplot, binary_classification_data, multiclass_classification_data
):
    """Check that we can pass keyword arguments to the precision-recall curve plot."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.plot.precision_recall()
    for pr_curve_kwargs in ({"color": "red"}, [{"color": "red"}]):
        display.plot(
            pr_curve_kwargs=pr_curve_kwargs,
            plot_chance_level=True,
            chance_level_kwargs={"color": "blue"},
        )

        assert display.lines_[0].get_color() == "red"
        assert display.chance_levels_[0].get_color() == "blue"

    display.plot(plot_chance_level=True)
    assert display.chance_levels_[0].get_color() == "k"

    display.plot(plot_chance_level=True, chance_level_kwargs=[{"color": "red"}])
    assert display.chance_levels_[0].get_color() == "red"

    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.plot.precision_recall()
    display.plot(
        pr_curve_kwargs=[dict(color="red"), dict(color="blue"), dict(color="green")],
        plot_chance_level=True,
        chance_level_kwargs=[
            dict(color="red"),
            dict(color="blue"),
            dict(color="green"),
        ],
    )
    assert display.lines_[0].get_color() == "red"
    assert display.lines_[1].get_color() == "blue"
    assert display.lines_[2].get_color() == "green"
    assert display.chance_levels_[0].get_color() == "red"
    assert display.chance_levels_[1].get_color() == "blue"
    assert display.chance_levels_[2].get_color() == "green"

    display.plot(plot_chance_level=True)
    for chance_level in display.chance_levels_:
        assert chance_level.get_color() == "k"

    display.plot(despine=False)
    assert display.ax_.spines["top"].get_visible()
    assert display.ax_.spines["right"].get_visible()


def test_precision_recall_curve_display_plot_error_wrong_pr_curve_kwargs(
    pyplot, binary_classification_data, multiclass_classification_data
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `roc_curve_kwargs` argument.
    """
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.plot.precision_recall()
    err_msg = (
        "You intend to plot a single precision-recall curve and provide multiple "
        "precision-recall curve keyword arguments"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs=[{}, {}])

    err_msg = (
        "You intend to plot a single chance level line and provide multiple chance "
        "level line keyword arguments"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(plot_chance_level=True, chance_level_kwargs=[{}, {}])

    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.plot.precision_recall()
    err_msg = "You intend to plot multiple precision-recall curves."
    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs=[{}, {}])

    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs={})

    err_msg = (
        "You intend to plot multiple precision-recall curves. We expect "
        "`chance_level_kwargs` to be a list"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(plot_chance_level=True, chance_level_kwargs=[{}, {}])

    with pytest.raises(ValueError, match=err_msg):
        display.plot(plot_chance_level=True, chance_level_kwargs={})
