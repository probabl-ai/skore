import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skore import ComparisonReport, CrossValidationReport, EstimatorReport
from skore.sklearn._plot import PrecisionRecallCurveDisplay
from skore.sklearn._plot.utils import sample_mpl_colormap


@pytest.fixture
def binary_classification_data():
    X, y = make_classification(class_sep=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return LogisticRegression().fit(X_train, y_train), X_train, X_test, y_train, y_test


@pytest.fixture
def multiclass_classification_data():
    X, y = make_classification(
        class_sep=0.1, n_classes=3, n_clusters_per_class=1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return LogisticRegression().fit(X_train, y_train), X_train, X_test, y_train, y_test


@pytest.fixture
def binary_classification_data_no_split():
    X, y = make_classification(random_state=42)
    return LogisticRegression(), X, y


@pytest.fixture
def multiclass_classification_data_no_split():
    X, y = make_classification(n_classes=3, n_clusters_per_class=1, random_state=42)
    return LogisticRegression(), X, y


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
    display = report.metrics.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)

    # check the structure of the attributes
    for attr_name in ("precision", "recall", "average_precision"):
        assert isinstance(getattr(display, attr_name), dict)
        assert len(getattr(display, attr_name)) == 1

        attr = getattr(display, attr_name)
        assert list(attr.keys()) == [estimator.classes_[1]]
        assert list(attr.keys()) == [display.pos_label]
        assert isinstance(attr[estimator.classes_[1]], list)
        assert len(attr[estimator.classes_[1]]) == 1

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == 1
    precision_recall_curve_mpl = display.lines_[0]
    assert isinstance(precision_recall_curve_mpl, mpl.lines.Line2D)
    assert (
        precision_recall_curve_mpl.get_label()
        == f"Test set (AP = {display.average_precision[estimator.classes_[1]][0]:0.2f})"
    )
    assert precision_recall_curve_mpl.get_color() == "#1f77b4"  # tab:blue in hex

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == estimator.__class__.__name__
    assert len(legend.get_texts()) == 1

    assert display.ax_.get_xlabel() == "Recall\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "Precision\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_precision_recall_curve_cross_validation_display_binary_classification(
    pyplot, binary_classification_data_no_split
):
    """Check the attributes and default plotting behaviour of the
    precision-recall curve plot with binary data.
    """
    (estimator, X, y), cv = binary_classification_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)

    # check the structure of the attributes
    pos_label = report.estimator_reports_[0].estimator_.classes_[1]
    for attr_name in ("precision", "recall", "average_precision"):
        assert isinstance(getattr(display, attr_name), dict)
        assert len(getattr(display, attr_name)) == 1

        attr = getattr(display, attr_name)
        assert list(attr.keys()) == [pos_label]
        assert list(attr.keys()) == [display.pos_label]
        assert isinstance(attr[pos_label], list)
        assert len(attr[pos_label]) == cv

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == cv
    expected_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for split_idx, line in enumerate(display.lines_):
        assert isinstance(line, mpl.lines.Line2D)
        assert line.get_label() == (
            f"Test set - fold #{split_idx + 1} "
            f"(AP = {display.average_precision[pos_label][split_idx]:0.2f})"
        )
        assert mpl.colors.to_rgba(line.get_color()) == expected_colors[split_idx]

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == estimator.__class__.__name__
    assert len(legend.get_texts()) == 3

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
    display = report.metrics.precision_recall(data_source="train")
    display.plot()
    assert display.lines_[0].get_label() == "Train set (AP = 0.93)"

    display = report.metrics.precision_recall(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    assert display.lines_[0].get_label() == "AP = 0.93"


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
    display = report.metrics.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)

    # check the structure of the attributes
    for attr_name in ("precision", "recall", "average_precision"):
        assert isinstance(getattr(display, attr_name), dict)
        assert len(getattr(display, attr_name)) == len(estimator.classes_)

        attr = getattr(display, attr_name)
        for class_label in estimator.classes_:
            assert isinstance(attr[class_label], list)
            assert len(attr[class_label]) == 1

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(estimator.classes_)
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for class_label, expected_color in zip(estimator.classes_, default_colors):
        precision_recall_curve_mpl = display.lines_[class_label]
        assert isinstance(precision_recall_curve_mpl, mpl.lines.Line2D)
        assert precision_recall_curve_mpl.get_label() == (
            f"{str(class_label).title()} - test set "
            f"(AP = {display.average_precision[class_label][0]:0.2f})"
        )
        assert precision_recall_curve_mpl.get_color() == expected_color

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == estimator.__class__.__name__
    assert len(legend.get_texts()) == 3

    assert display.ax_.get_xlabel() == "Recall"
    assert display.ax_.get_ylabel() == "Precision"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_precision_recall_curve_cross_validation_display_multiclass_classification(
    pyplot, multiclass_classification_data_no_split
):
    """Check the attributes and default plotting behaviour of the precision-recall
    curve plot with multiclass data.
    """
    (estimator, X, y), cv = multiclass_classification_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)

    # check the structure of the attributes
    class_labels = report.estimator_reports_[0].estimator_.classes_
    for attr_name in ("precision", "recall", "average_precision"):
        assert isinstance(getattr(display, attr_name), dict)
        assert len(getattr(display, attr_name)) == len(class_labels)

        attr = getattr(display, attr_name)
        for class_label in class_labels:
            assert isinstance(attr[class_label], list)
            assert len(attr[class_label]) == cv

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(class_labels) * cv
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for class_label, expected_color in zip(class_labels, default_colors):
        for split_idx in range(cv):
            roc_curve_mpl = display.lines_[class_label * cv + split_idx]
            assert isinstance(roc_curve_mpl, mpl.lines.Line2D)
            if split_idx == 0:
                assert roc_curve_mpl.get_label() == (
                    f"{str(class_label).title()} - test set "
                    f"(AP = {np.mean(display.average_precision[class_label]):0.2f}"
                    f" +/- {np.std(display.average_precision[class_label]):0.2f})"
                )
            assert roc_curve_mpl.get_color() == expected_color

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
    for pr_curve_kwargs in ({"color": "red"}, [{"color": "red"}]):
        display = report.metrics.precision_recall()
        display.plot(pr_curve_kwargs=pr_curve_kwargs)
        assert display.lines_[0].get_color() == "red"

        # check the `.style` display setter
        display.plot()  # default style
        assert display.lines_[0].get_color() == "#1f77b4"
        display.set_style(pr_curve_kwargs=pr_curve_kwargs)
        display.plot()
        assert display.lines_[0].get_color() == "red"
        display.plot(pr_curve_kwargs=pr_curve_kwargs)
        assert display.lines_[0].get_color() == "red"

        # reset to default style since next call to `precision_recall` will use the
        # cache
        display.set_style(pr_curve_kwargs={"color": "#1f77b4"})

    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()
    display.plot(
        pr_curve_kwargs=[dict(color="red"), dict(color="blue"), dict(color="green")],
    )
    assert display.lines_[0].get_color() == "red"
    assert display.lines_[1].get_color() == "blue"
    assert display.lines_[2].get_color() == "green"

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
    display = report.metrics.precision_recall()
    err_msg = (
        "You intend to plot a single curve. We expect `pr_curve_kwargs` to be a "
        "dictionary."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs=[{}, {}])

    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall()
    err_msg = (
        "You intend to plot multiple curves. We expect `pr_curve_kwargs` to be a "
        "list of dictionaries."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs=[{}, {}])

    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs={})


@pytest.mark.parametrize(
    "fixture_name",
    ["binary_classification_data_no_split", "multiclass_classification_data_no_split"],
)
@pytest.mark.parametrize("pr_curve_kwargs", [[{"color": "red"}], "unknown"])
def test_pr_curve_display_cross_validation_multiple_roc_curve_kwargs_error(
    pyplot, fixture_name, request, pr_curve_kwargs
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `pr_curve_kwargs` argument."""
    (estimator, X, y), cv = request.getfixturevalue(fixture_name), 3

    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.precision_recall()
    err_msg = (
        "You intend to plot multiple curves. We expect `pr_curve_kwargs` to be a list "
        "of dictionaries"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs=pr_curve_kwargs)


def test_precision_recall_curve_display_data_source_binary_classification(
    pyplot, binary_classification_data
):
    """Check that we can pass the `data_source` argument to the precision-recall curve
    plot."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall(data_source="train")
    display.plot()
    assert display.lines_[0].get_label() == "Train set (AP = 0.93)"

    display = report.metrics.precision_recall(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    assert display.lines_[0].get_label() == "AP = 0.93"


def test_precision_recall_curve_display_data_source_multiclass_classification(
    pyplot, multiclass_classification_data
):
    """Check that we can pass the `data_source` argument to the precision-recall curve
    plot."""
    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.precision_recall(data_source="train")
    display.plot()
    for class_label in estimator.classes_:
        assert display.lines_[class_label].get_label() == (
            f"{str(class_label).title()} - train set "
            f"(AP = {display.average_precision[class_label][0]:0.2f})"
        )

    display = report.metrics.precision_recall(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    for class_label in estimator.classes_:
        assert display.lines_[class_label].get_label() == (
            f"{str(class_label).title()} - "
            f"AP = {display.average_precision[class_label][0]:0.2f}"
        )


def test_precision_recall_curve_display_comparison_report_binary_classification(
    pyplot, binary_classification_data
):
    """Check the attributes and default plotting behaviour of the precision-recall curve
    plot with binary data."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    estimator_2 = clone(estimator).set_params(C=10).fit(X_train, y_train)
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)

    # check the structure of the attributes
    for attr_name in ("precision", "recall", "average_precision"):
        assert isinstance(getattr(display, attr_name), dict)
        assert len(getattr(display, attr_name)) == 1

        attr = getattr(display, attr_name)
        assert list(attr.keys()) == [estimator.classes_[1]]
        assert list(attr.keys()) == [display.pos_label]
        assert isinstance(attr[estimator.classes_[1]], list)
        assert len(attr[estimator.classes_[1]]) == 2

    display.plot()
    expected_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for split_idx, line in enumerate(display.lines_):
        assert isinstance(line, mpl.lines.Line2D)
        assert line.get_label() == (
            f"{report.report_names_[split_idx]} "
            f"(AP = {display.average_precision[display.pos_label][split_idx]:0.2f})"
        )
        assert mpl.colors.to_rgba(line.get_color()) == expected_colors[split_idx]

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == r"Binary-Classification on $\bf{test}$ set"
    assert len(legend.get_texts()) == 2

    assert display.ax_.get_xlabel() == "Recall\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "Precision\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_precision_recall_curve_display_comparison_report_multiclass_classification(
    pyplot, multiclass_classification_data
):
    """Check the attributes and default plotting behaviour of the precision-recall curve
    plot with multiclass data."""
    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    estimator_2 = clone(estimator).set_params(C=10).fit(X_train, y_train)
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)

    # check the structure of the attributes
    class_labels = report.estimator_reports_[0].estimator_.classes_
    for attr_name in ("precision", "recall", "average_precision"):
        assert isinstance(getattr(display, attr_name), dict)
        assert len(getattr(display, attr_name)) == len(class_labels)

        attr = getattr(display, attr_name)
        for class_label in class_labels:
            assert isinstance(attr[class_label], list)
            assert len(attr[class_label]) == 2

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(class_labels) * 2
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for estimator_idx, expected_color in zip(
        range(len(report.report_names_)), default_colors
    ):
        for class_label_idx, class_label in enumerate(class_labels):
            roc_curve_mpl = display.lines_[
                estimator_idx * len(class_labels) + class_label_idx
            ]
            assert isinstance(roc_curve_mpl, mpl.lines.Line2D)
            assert roc_curve_mpl.get_label() == (
                f"{report.report_names_[estimator_idx]} - {str(class_label).title()} "
                f"(AP = {display.average_precision[class_label][estimator_idx]:0.2f})"
            )
            assert roc_curve_mpl.get_color() == expected_color

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert (
        legend.get_title().get_text() == r"Multiclass-Classification on $\bf{test}$ set"
    )
    assert len(legend.get_texts()) == 6

    assert display.ax_.get_xlabel() == "Recall"
    assert display.ax_.get_ylabel() == "Precision"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_precision_recall_curve_display_comparison_report_binary_classification_kwargs(
    pyplot, binary_classification_data
):
    """Check that we can pass keyword arguments to the precision-recall curve plot for
    cross-validation."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    estimator_2 = clone(estimator).set_params(C=10).fit(X_train, y_train)
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.precision_recall()
    pr_curve_kwargs = [{"color": "red"}, {"color": "blue"}]
    display.plot(pr_curve_kwargs=pr_curve_kwargs)
    assert display.lines_[0].get_color() == "red"
    assert display.lines_[1].get_color() == "blue"


@pytest.mark.parametrize(
    "fixture_name",
    ["binary_classification_data", "multiclass_classification_data"],
)
@pytest.mark.parametrize("pr_curve_kwargs", [[{"color": "red"}], "unknown"])
def test_pr_curve_display_comparison_multiple_pr_curve_kwargs_error(
    pyplot, fixture_name, request, pr_curve_kwargs
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `pr_curve_kwargs` argument."""
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)

    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.precision_recall()
    err_msg = "You intend to plot multiple curves"
    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs=pr_curve_kwargs)


def test_precision_recall_curve_display_wrong_report_type(
    pyplot, binary_classification_data
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `report_type` argument."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    estimator_report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = estimator_report.metrics.precision_recall()
    display.report_type = "unknown"
    err_msg = (
        "`report_type` should be one of 'estimator', 'cross-validation', "
        "or 'comparison-estimator'. Got 'unknown' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot()
