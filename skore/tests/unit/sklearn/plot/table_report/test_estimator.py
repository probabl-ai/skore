import pandas as pd
import pytest
from matplotlib.collections import QuadMesh
from sklearn.model_selection import train_test_split
from skore import Display, EstimatorReport
from skrub import tabular_learner
from skrub.datasets import fetch_employee_salaries


@pytest.fixture
def skrub_data():
    data = fetch_employee_salaries()
    return train_test_split(data.X, data.y, random_state=0)


@pytest.fixture
def estimator_report():
    data = fetch_employee_salaries()
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, random_state=0)
    return EstimatorReport(
        tabular_learner("regressor"),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def test_table_report_display_constructor(estimator_report):
    """Check the value that are stored in the display constructor."""
    display = estimator_report.data.analyze()
    assert isinstance(display, Display)

    assert hasattr(display, "summary")
    assert isinstance(display.summary, dict)
    assert list(display.summary.keys()) == [
        "dataframe",
        "dataframe_module",
        "n_rows",
        "n_columns",
        "columns",
        "dataframe_is_empty",
        "plots_skipped",
        "sample_table",
        "n_constant_columns",
        "top_associations",
    ]


def test_table_report_display_frame_error(estimator_report):
    """Check the error message when passing an unknown kind in the `frame` method."""
    display = estimator_report.data.analyze()
    with pytest.raises(ValueError, match="Invalid kind: 'xxx'"):
        display.frame(kind="xxx")


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_table_report_display_frame(estimator_report, data_source):
    display = estimator_report.data.analyze(data_source=data_source)
    dataset = display.frame(kind="dataset")

    if data_source == "train":
        pd.testing.assert_frame_equal(
            dataset,
            pd.concat([estimator_report.X_train, estimator_report.y_train], axis=1),
        )
    else:
        pd.testing.assert_frame_equal(
            dataset,
            pd.concat([estimator_report.X_test, estimator_report.y_test], axis=1),
        )

    associations = display.frame(kind="top-associations")
    pd.testing.assert_frame_equal(
        associations, pd.DataFrame(display.summary["top_associations"])
    )


def test_simple_categ_plots_1d(pyplot, estimator_report):
    display = estimator_report.data.analyze(data_source="train")
    display.plot(x="gender")
    assert hasattr(display, "ax_")
    assert hasattr(display, "figure_")
    assert display.ax_.get_xlabel() == "gender"
    assert [label.get_text() for label in display.ax_.get_xticklabels()] == ["M", "F"]
    labels = display.ax_.get_yticklabels()
    assert labels[0].get_text() == "0"
    assert labels[-1].get_text() == "4000"
    assert display.ax_.get_ylabel() == "Count"
    # orange
    assert display.ax_.containers[0].patches[0].get_facecolor() == (
        1.0,
        0.4980392156862745,
        0.054901960784313725,
        0.75,
    )

    display.plot(y="gender", histplot_kwargs={"color": "blue"})
    assert display.ax_.get_xlabel() == "Count"
    assert display.ax_.get_ylabel() == "gender"
    # blue
    assert display.ax_.containers[0].patches[0].get_facecolor() == (0.0, 0.0, 1.0, 0.75)


def test_simple_num_plots_1d(pyplot, estimator_report):
    display = estimator_report.data.analyze(data_source="train")
    display.plot(x="year_first_hired", histplot_kwargs={"color": "red"})
    assert display.ax_.get_xlabel() == "year_first_hired"
    labels = display.ax_.get_xticklabels()
    assert labels[0].get_text() == "1970"
    assert labels[-1].get_text() == "2010"
    labels = display.ax_.get_yticklabels()
    assert labels[0].get_text() == "0"
    assert labels[-1].get_text() == "500"
    assert display.ax_.get_ylabel() == "Count"
    # red
    assert display.ax_.containers[0].patches[0].get_facecolor() == (1.0, 0.0, 0.0, 0.75)

    display.plot(y="year_first_hired")
    assert display.ax_.get_xlabel() == "Count"
    assert display.ax_.get_ylabel() == "year_first_hired"


def test_top_k_categ_plots_1d(pyplot, estimator_report):
    display = estimator_report.data.analyze(data_source="train")
    display.plot(x="division")
    assert len(display.ax_.get_xticklabels()) == 20
    display.plot(x="division", top_k_categories=30)
    assert len(display.ax_.get_xticklabels()) == 30


def test_hue_plots_1d(pyplot, estimator_report):
    display = estimator_report.data.analyze(data_source="train")
    display.plot(x="gender", hue="current_annual_salary")
    assert "BoxPlotContainer" in display.ax_.containers[0].__class__.__name__
    legend_labels = display.ax_.legend_.texts
    assert legend_labels[0].get_text() == "50000"
    assert legend_labels[-1].get_text() == "300000"
    assert display.ax_.legend_.get_title().get_text() == "current_annual_salary"

    display.plot(y="gender", hue="current_annual_salary")
    assert "BoxPlotContainer" in display.ax_.containers[0].__class__.__name__

    msg = "If 'x' and 'y' are categories, 'hue' must be continuous"
    with pytest.raises(ValueError, match=msg):
        display.plot(x="gender", hue="division", top_k_categories=30)

    display.plot(y="year_first_hired", hue="current_annual_salary")
    assert display.ax_.get_xlabel() == "current_annual_salary"
    assert display.ax_.get_ylabel() == "year_first_hired"
    assert display.ax_.legend_.get_title().get_text() == "current_annual_salary"


def test_simple_plots_2d(pyplot, estimator_report):
    display = estimator_report.data.analyze(data_source="train")
    # scatter plot
    display.plot(y="current_annual_salary", x="year_first_hired")
    assert display.ax_.get_xlabel() == "year_first_hired"
    assert display.ax_.get_ylabel() == "current_annual_salary"
    labels = display.ax_.get_xticklabels()
    assert labels[0].get_text() == "1970"
    assert labels[-1].get_text() == "2010"
    labels = display.ax_.get_yticklabels()
    assert labels[0].get_text() == "0"
    assert labels[-1].get_text() == "300000"

    # box plot
    display.plot(x="current_annual_salary", y="division")
    assert display.ax_.get_ylabel() == "division"
    assert display.ax_.get_xlabel() == "current_annual_salary"
    assert len(display.ax_.lines) == 154

    # heatmap
    display.plot(x="gender", y="division")
    assert len(display.ax_.get_yticklabels()) == 19
    assert display.ax_.get_ylabel() == "division"
    assert display.ax_.get_xlabel() == "gender"
    assert isinstance(display.ax_.collections[0], QuadMesh)


def test_hue_plots_2d(pyplot, estimator_report):
    display = estimator_report.data.analyze(data_source="train")
    display.plot(x="year_first_hired", y="current_annual_salary", hue="division")
    assert len(display.ax_.legend_.texts) == 22
    assert display.ax_.legend_.get_title().get_text() == "division"

    display.plot(x="year_first_hired", y="gender", hue="division")
    assert len(display.ax_.lines) == 43
    assert len(display.ax_.legend_.texts) == 22
    assert display.ax_.legend_.get_title().get_text() == "division"

    display.plot(x="gender", y="division", hue="current_annual_salary")
    assert isinstance(display.ax_.collections[0], QuadMesh)
    colorbar = display.ax_.collections[0].colorbar
    assert colorbar.vmin == pytest.approx(28813.63, rel=1e-1)
    assert colorbar.vmax == pytest.approx(82980.51, rel=1e-1)

    # Can't have categorical hue when x and y are categories
    msg = "If 'x' and 'y' are categories, 'hue' must be continuous."
    with pytest.raises(ValueError, match=msg):
        display.plot(x="gender", y="division", hue="department_name")


def test_corr_plot(pyplot, estimator_report):
    display = estimator_report.data.analyze(data_source="train")
    display.plot(kind="corr")
    assert isinstance(display.ax_.collections[0], QuadMesh)
    assert len(display.ax_.get_xticklabels()) == 9
    assert len(display.ax_.get_yticklabels()) == 9
    assert display.ax_.title.get_text() == "Cramer's V Correlation"
