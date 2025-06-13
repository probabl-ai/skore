import matplotlib as mpl
import polars as pl
import pytest
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore.sklearn._plot.data import TableReportDisplay
from skrub import _dataframe as sbd
from skrub import tabular_learner
from skrub._reporting._summarize import summarize_dataframe
from skrub.datasets import fetch_employee_salaries

try:
    import polars as pl

    # Ensure pl.from_pandas is available
    import pyarrow  # noqa: F401

    _POLARS_INSTALLED = True
except ImportError:
    _POLARS_INSTALLED = False


@pytest.fixture
def skrub_data(request):
    data = fetch_employee_salaries()
    if request.param == "polars":
        data.X = pl.from_pandas(data.X)
        data.y = pl.from_pandas(data.y)
    return train_test_split(data.X, data.y, random_state=0)


@pytest.mark.parametrize(
    "skrub_data",
    [
        "pandas",
        pytest.param(
            "polars",
            marks=pytest.mark.skipif(
                not _POLARS_INSTALLED, reason="Polars not installed"
            ),
        ),
    ],
    indirect=True,
)
def test_distribution_plot(pyplot, skrub_data):
    X_train, X_test, y_train, y_test = skrub_data
    n_train, n_test = sbd.shape(X_train)[0], sbd.shape(X_test)[0]
    report = EstimatorReport(
        tabular_learner("regressor"),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    for source_dataset, n_samples in zip(
        ["train", "test", "all"],
        [n_train, n_test, n_train + n_test],
        strict=False,
    ):
        display = report.data.analyze(source_dataset=source_dataset, with_y=True)
        assert sbd.shape(display.df)[0] == n_samples

    display = report.data.analyze(source_dataset="train", with_y=True)
    assert isinstance(display, TableReportDisplay)
    summary_1, summary_2 = display.summary, summarize_dataframe(X_train)
    assert summary_1["n_rows"] == summary_2["n_rows"]
    assert (
        summary_1["n_columns"] == summary_2["n_columns"] + 1
    )  # +1 for the target column

    # Test distribution plot for a categorical column
    display.plot(x="gender")
    assert isinstance(display.ax_, mpl.axes.Axes)

    # Test distribution plot for a numerical column
    display.plot(x="current_annual_salary")

    # Test distribution plot for a numerical column with y-axis
    display.plot(
        y="current_annual_salary",
        x="year_first_hired",
    )

    # Test distribution plot for a numerical column with y-axis and hue
    display.plot(
        y="current_annual_salary",
        x="year_first_hired",
        hue="current_annual_salary",
    )

    # Test distribution plot for a categorical column with top_k_categories
    display.plot(
        x="division",
        y="current_annual_salary",
        top_k_categories=5,
    )

    # Test distribution plot for a categorical column with top_k_categories and y-axis
    display.plot(
        x="division",
        y="current_annual_salary",
        top_k_categories=5,
    )

    # Test Cramer's V correlation plot
    display.plot(
        kind="cramer", heatmap_kwargs={"xticklabels": False, "cmap": "viridis"}
    )
