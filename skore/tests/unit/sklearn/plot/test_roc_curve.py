import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skore import ComparisonReport, CrossValidationReport, EstimatorReport
from skore.sklearn._plot import RocCurveDisplay
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


def get_roc_auc(
    display,
    label=None,
    split_number=None,
    estimator_name=None,
) -> float:
    noop_filter = display.roc_auc["roc_auc"].map(lambda x: True)
    label_filter = (display.roc_auc["label"] == label) if label is not None else True
    split_number_filter = (
        (display.roc_auc["split_index"] == split_number)
        if split_number is not None
        else True
    )
    estimator_name_filter = (
        (display.roc_auc["estimator_name"] == estimator_name)
        if estimator_name is not None
        else True
    )
    return display.roc_auc[
        noop_filter & label_filter & split_number_filter & estimator_name_filter
    ]["roc_auc"].iloc[0]
    # return display.roc_auc[pos_label][split_idx]


def test_roc_curve_display_binary_classification(pyplot, binary_classification_data):
    """Check the attributes and default plotting behaviour of the ROC curve plot with
    binary data."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    assert isinstance(display, RocCurveDisplay)

    # check the structure of the attributes
    assert list(display.roc_curve.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "threshold",
        "fpr",
        "tpr",
    ]
    assert list(display.roc_auc.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "roc_auc",
    ]

    assert (
        list(display.roc_curve["label"].unique())
        == list(display.roc_auc["label"].unique())
        == [estimator.classes_[1]]
        == [display.pos_label]
    )

    display.plot()
    assert hasattr(display, "ax_")
    assert hasattr(display, "figure_")
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == 1
    roc_curve_mpl = display.lines_[0]
    assert isinstance(roc_curve_mpl, mpl.lines.Line2D)
    assert roc_curve_mpl.get_label() == f"Test set (AUC = {get_roc_auc(display):0.2f})"
    assert roc_curve_mpl.get_color() == "#1f77b4"  # tab:blue in hex

    assert isinstance(display.chance_level_, mpl.lines.Line2D)
    assert display.chance_level_.get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_.get_color() == "k"

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == estimator.__class__.__name__
    assert len(legend.get_texts()) == 1 + 1

    assert display.ax_.get_xlabel() == "False Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "True Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_roc_curve_display_multiclass_classification(
    pyplot, multiclass_classification_data
):
    """Check the attributes and default plotting behaviour of the ROC curve plot with
    multiclass data."""
    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    assert isinstance(display, RocCurveDisplay)

    # check the structure of the attributes
    assert list(display.roc_curve.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "threshold",
        "fpr",
        "tpr",
    ]
    assert list(display.roc_auc.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "roc_auc",
    ]

    np.testing.assert_array_equal(
        display.roc_curve["label"].unique(), estimator.classes_
    )
    np.testing.assert_array_equal(display.roc_auc["label"].unique(), estimator.classes_)

    display.plot()
    assert hasattr(display, "ax_")
    assert hasattr(display, "figure_")
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(estimator.classes_)
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for class_label, expected_color in zip(estimator.classes_, default_colors):
        roc_curve_mpl = display.lines_[class_label]
        assert isinstance(roc_curve_mpl, mpl.lines.Line2D)
        roc_auc_class = get_roc_auc(display, label=class_label)
        assert roc_curve_mpl.get_label() == (
            f"{str(class_label).title()} - test set (AUC = {roc_auc_class:0.2f})"
        )
        assert roc_curve_mpl.get_color() == expected_color

    assert isinstance(display.chance_level_, mpl.lines.Line2D)
    assert display.chance_level_.get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_.get_color() == "k"

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == estimator.__class__.__name__
    assert len(legend.get_texts()) == len(estimator.classes_) + 1

    assert display.ax_.get_xlabel() == "False Positive Rate"
    assert display.ax_.get_ylabel() == "True Positive Rate"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_roc_curve_display_data_source_binary_classification(
    pyplot, binary_classification_data
):
    """Check that we can pass the `data_source` argument to the ROC curve plot."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc(data_source="train")
    display.plot()
    assert (
        display.lines_[0].get_label()
        == f"Train set (AUC = {get_roc_auc(display):0.2f})"
    )

    display = report.metrics.roc(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    assert display.lines_[0].get_label() == f"AUC = {get_roc_auc(display):0.2f}"


def test_roc_curve_display_data_source_multiclass_classification(
    pyplot, multiclass_classification_data
):
    """Check that we can pass the `data_source` argument to the ROC curve plot."""
    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc(data_source="train")
    display.plot()
    for class_label in estimator.classes_:
        assert display.lines_[class_label].get_label() == (
            f"{str(class_label).title()} - train set "
            f"(AUC = {get_roc_auc(display, label=class_label):0.2f})"
        )

    display = report.metrics.roc(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    for class_label in estimator.classes_:
        assert display.lines_[class_label].get_label() == (
            f"{str(class_label).title()} - "
            f"AUC = {get_roc_auc(display, label=class_label):0.2f}"
        )


def test_roc_curve_display_plot_error_wrong_roc_curve_kwargs(
    pyplot, binary_classification_data, multiclass_classification_data
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `roc_curve_kwargs` argument."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    err_msg = (
        "You intend to plot a single curve. We expect `roc_curve_kwargs` to be a "
        "dictionary."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(roc_curve_kwargs=[{}, {}])

    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    err_msg = "You intend to plot multiple curves."
    with pytest.raises(ValueError, match=err_msg):
        display.plot(roc_curve_kwargs=[{}, {}])

    with pytest.raises(ValueError, match=err_msg):
        display.plot(roc_curve_kwargs={})


@pytest.mark.parametrize("roc_curve_kwargs", [[{"color": "red"}], {"color": "red"}])
def test_roc_curve_display_roc_curve_kwargs_binary_classification(
    pyplot, binary_classification_data, roc_curve_kwargs
):
    """Check that we can pass keyword arguments to the ROC curve plot."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    display.plot(
        roc_curve_kwargs=roc_curve_kwargs, chance_level_kwargs={"color": "blue"}
    )

    assert display.lines_[0].get_color() == "red"
    assert display.chance_level_.get_color() == "blue"

    # check the `.style` display setter
    display.plot()  # default style
    assert display.lines_[0].get_color() == "#1f77b4"
    assert display.chance_level_.get_color() == "k"
    display.set_style(
        roc_curve_kwargs=roc_curve_kwargs, chance_level_kwargs={"color": "blue"}
    )
    display.plot()
    assert display.lines_[0].get_color() == "red"
    assert display.chance_level_.get_color() == "blue"
    # overwrite the style that was set above
    display.plot(
        roc_curve_kwargs={"color": "#1f77b4"}, chance_level_kwargs={"color": "red"}
    )
    assert display.lines_[0].get_color() == "#1f77b4"
    assert display.chance_level_.get_color() == "red"


def test_roc_curve_display_roc_curve_kwargs_multiclass_classification(
    pyplot, multiclass_classification_data
):
    """Check that we can pass keyword arguments to the ROC curve plot for
    multiclass classification."""
    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    display.plot(
        roc_curve_kwargs=[dict(color="red"), dict(color="blue"), dict(color="green")],
        chance_level_kwargs={"color": "blue"},
    )
    assert display.lines_[0].get_color() == "red"
    assert display.lines_[1].get_color() == "blue"
    assert display.lines_[2].get_color() == "green"
    assert display.chance_level_.get_color() == "blue"

    display.plot(plot_chance_level=False)
    assert display.chance_level_ is None

    display.plot(despine=False)
    assert display.ax_.spines["top"].get_visible()
    assert display.ax_.spines["right"].get_visible()


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_roc_curve_display_cross_validation_binary_classification(
    pyplot, binary_classification_data_no_split, data_source
):
    """Check the attributes and default plotting behaviour of the ROC curve plot with
    binary data."""
    (estimator, X, y), cv = binary_classification_data_no_split, 3
    if data_source == "X_y":
        roc_kwargs = {"data_source": data_source, "X": X, "y": y}
    else:
        roc_kwargs = {"data_source": data_source}

    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.roc(**roc_kwargs)
    assert isinstance(display, RocCurveDisplay)

    # check the structure of the attributes
    pos_label = report.estimator_reports_[0].estimator_.classes_[1]

    assert list(display.roc_curve.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "threshold",
        "fpr",
        "tpr",
    ]
    assert list(display.roc_auc.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "roc_auc",
    ]

    assert (
        list(display.roc_curve["label"].unique())
        == list(display.roc_auc["label"].unique())
        == [pos_label]
        == [display.pos_label]
    )
    assert (
        len(display.roc_curve["split_index"].unique())
        == len(display.roc_auc["split_index"].unique())
        == cv
    )

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == cv
    expected_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for split_idx, line in enumerate(display.lines_):
        assert isinstance(line, mpl.lines.Line2D)
        roc_auc_split = get_roc_auc(display, label=pos_label, split_number=split_idx)
        assert line.get_label() == (
            f"Estimator of fold #{split_idx + 1} (AUC = {roc_auc_split:0.2f})"
        )
        assert mpl.colors.to_rgba(line.get_color()) == expected_colors[split_idx]

    assert isinstance(display.chance_level_, mpl.lines.Line2D)
    assert display.chance_level_.get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_.get_color() == "k"

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    data_source_title = "external" if data_source == "X_y" else data_source
    assert (
        legend.get_title().get_text()
        == f"LogisticRegression on $\\bf{{{data_source_title}}}$ set"
    )
    assert len(legend.get_texts()) == cv + 1

    assert display.ax_.get_xlabel() == "False Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "True Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_roc_curve_display_cross_validation_multiclass_classification(
    pyplot, multiclass_classification_data_no_split, data_source
):
    """Check the attributes and default plotting behaviour of the ROC curve plot with
    multiclass data."""
    (estimator, X, y), cv = multiclass_classification_data_no_split, 3
    if data_source == "X_y":
        roc_kwargs = {"data_source": data_source, "X": X, "y": y}
    else:
        roc_kwargs = {"data_source": data_source}

    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.roc(**roc_kwargs)
    assert isinstance(display, RocCurveDisplay)

    # check the structure of the attributes
    assert list(display.roc_curve.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "threshold",
        "fpr",
        "tpr",
    ]
    assert list(display.roc_auc.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "roc_auc",
    ]

    class_labels = report.estimator_reports_[0].estimator_.classes_
    assert (
        list(display.roc_curve["label"].unique())
        == list(display.roc_auc["label"].unique())
        == list(class_labels)
    )
    assert (
        len(display.roc_curve["split_index"].unique())
        == len(display.roc_auc["split_index"].unique())
        == cv
    )

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(class_labels) * cv
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for class_label, expected_color in zip(class_labels, default_colors):
        for split_idx in range(cv):
            roc_curve_mpl = display.lines_[class_label * cv + split_idx]
            assert isinstance(roc_curve_mpl, mpl.lines.Line2D)
            if split_idx == 0:
                roc_auc_class = get_roc_auc(display, label=class_label)
                assert roc_curve_mpl.get_label() == (
                    f"{str(class_label).title()} "
                    f"(AUC = {np.mean(roc_auc_class):0.2f}"
                    f" +/- {np.std(roc_auc_class):0.2f})"
                )
            assert roc_curve_mpl.get_color() == expected_color

    assert isinstance(display.chance_level_, mpl.lines.Line2D)
    assert display.chance_level_.get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_.get_color() == "k"

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    data_source_title = "external" if data_source == "X_y" else data_source
    assert (
        legend.get_title().get_text()
        == f"LogisticRegression on $\\bf{{{data_source_title}}}$ set"
    )
    assert len(legend.get_texts()) == cv + 1

    assert display.ax_.get_xlabel() == "False Positive Rate"
    assert display.ax_.get_ylabel() == "True Positive Rate"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


@pytest.mark.parametrize(
    "roc_curve_kwargs",
    [
        [{"color": "red"}, {"color": "blue"}, {"color": "green"}],
        {"color": "red"},
    ],
)
def test_roc_curve_display_cross_validation_binary_classification_kwargs(
    pyplot, binary_classification_data_no_split, roc_curve_kwargs
):
    """Check that we can pass keyword arguments to the ROC curve plot for
    cross-validation."""
    (estimator, X, y), cv = binary_classification_data_no_split, 3

    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.roc()
    display.plot(roc_curve_kwargs=roc_curve_kwargs)
    if isinstance(roc_curve_kwargs, list):
        assert display.lines_[0].get_color() == "red"
        assert display.lines_[1].get_color() == "blue"
        assert display.lines_[2].get_color() == "green"
    else:
        for line in display.lines_:
            assert line.get_color() == "red"


@pytest.mark.parametrize(
    "fixture_name",
    ["binary_classification_data_no_split", "multiclass_classification_data_no_split"],
)
@pytest.mark.parametrize("roc_curve_kwargs", [[{"color": "red"}], "unknown"])
def test_roc_curve_display_cross_validation_multiple_roc_curve_kwargs_error(
    pyplot, fixture_name, request, roc_curve_kwargs
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `roc_curve_kwargs` argument."""
    (estimator, X, y), cv = request.getfixturevalue(fixture_name), 3

    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.roc()
    err_msg = "You intend to plot multiple curves"
    with pytest.raises(ValueError, match=err_msg):
        display.plot(roc_curve_kwargs=roc_curve_kwargs)


def test_roc_curve_display_comparison_report_binary_classification(
    pyplot, binary_classification_data
):
    """Check the attributes and default plotting behaviour of the ROC curve plot with
    binary data."""
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
    display = report.metrics.roc()
    assert isinstance(display, RocCurveDisplay)

    # check the structure of the attributes
    assert list(display.roc_curve.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "threshold",
        "fpr",
        "tpr",
    ]
    assert list(display.roc_auc.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "roc_auc",
    ]

    assert (
        list(display.roc_curve["label"].unique())
        == list(display.roc_auc["label"].unique())
        == [estimator.classes_[1]]
        == [display.pos_label]
    )

    display.plot()
    expected_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for idx, (estimator_name, line) in enumerate(
        zip(report.report_names_, display.lines_)
    ):
        assert isinstance(line, mpl.lines.Line2D)
        roc_auc_class = get_roc_auc(
            display, label=display.pos_label, estimator_name=estimator_name
        )
        assert line.get_label() == (f"{estimator_name} (AUC = {roc_auc_class:0.2f})")
        assert mpl.colors.to_rgba(line.get_color()) == expected_colors[idx]

    assert isinstance(display.chance_level_, mpl.lines.Line2D)
    assert display.chance_level_.get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_.get_color() == "k"

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == r"Binary-Classification on $\bf{test}$ set"
    assert len(legend.get_texts()) == 2 + 1

    assert display.ax_.get_xlabel() == "False Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "True Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_roc_curve_display_comparison_report_multiclass_classification(
    pyplot, multiclass_classification_data
):
    """Check the attributes and default plotting behaviour of the ROC curve plot with
    multiclass data."""
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
    display = report.metrics.roc()
    assert isinstance(display, RocCurveDisplay)

    # check the structure of the attributes
    class_labels = report.reports_[0].estimator_.classes_
    assert list(display.roc_curve.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "threshold",
        "fpr",
        "tpr",
    ]
    assert list(display.roc_auc.columns) == [
        "estimator_name",
        "split_index",
        "label",
        "roc_auc",
    ]

    assert (
        list(display.roc_curve["label"].unique())
        == list(display.roc_auc["label"].unique())
        == list(class_labels)
    )

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(class_labels) * 2
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for idx, (estimator_name, expected_color) in enumerate(
        zip(report.report_names_, default_colors)
    ):
        for class_label_idx, class_label in enumerate(class_labels):
            roc_curve_mpl = display.lines_[idx * len(class_labels) + class_label_idx]
            assert isinstance(roc_curve_mpl, mpl.lines.Line2D)
            roc_auc_class = get_roc_auc(
                display,
                label=class_label,
                estimator_name=estimator_name,
            )
            assert roc_curve_mpl.get_label() == (
                f"{estimator_name} - {str(class_label).title()} "
                f"(AUC = {roc_auc_class:0.2f})"
            )
            assert roc_curve_mpl.get_color() == expected_color

    assert isinstance(display.chance_level_, mpl.lines.Line2D)
    assert display.chance_level_.get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_.get_color() == "k"

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert (
        legend.get_title().get_text() == r"Multiclass-Classification on $\bf{test}$ set"
    )
    assert len(legend.get_texts()) == 6 + 1

    assert display.ax_.get_xlabel() == "False Positive Rate"
    assert display.ax_.get_ylabel() == "True Positive Rate"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


def test_roc_curve_display_comparison_report_binary_classification_kwargs(
    pyplot, binary_classification_data
):
    """Check that we can pass keyword arguments to the ROC curve plot for
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
    display = report.metrics.roc()
    roc_curve_kwargs = [{"color": "red"}, {"color": "blue"}]
    display.plot(roc_curve_kwargs=roc_curve_kwargs)
    assert display.lines_[0].get_color() == "red"
    assert display.lines_[1].get_color() == "blue"


@pytest.mark.parametrize(
    "fixture_name",
    ["binary_classification_data", "multiclass_classification_data"],
)
@pytest.mark.parametrize("roc_curve_kwargs", [[{"color": "red"}], "unknown"])
def test_roc_curve_display_comparison_multiple_roc_curve_kwargs_error(
    pyplot, fixture_name, request, roc_curve_kwargs
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `roc_curve_kwargs` argument."""
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
    display = report.metrics.roc()
    err_msg = "You intend to plot multiple curves"
    with pytest.raises(ValueError, match=err_msg):
        display.plot(roc_curve_kwargs=roc_curve_kwargs)


def test_roc_curve_display_wrong_report_type(pyplot, binary_classification_data):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `report_type` argument."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    estimator_report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = estimator_report.metrics.roc()
    display.report_type = "unknown"
    err_msg = (
        "`report_type` should be one of 'estimator', 'cross-validation', "
        "or 'comparison-estimator'. Got 'unknown' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot()
