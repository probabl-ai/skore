import polars as pl
import pytest
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skrub import _dataframe as sbd
from skrub import tabular_learner
from skrub.datasets import fetch_employee_salaries

try:
    import polars as pl
    import polars.testing

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


@pytest.mark.parametrize("skrub_data", ["pandas", "polars"], indirect=True)
def test_distribution_plot(skrub_data):
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
