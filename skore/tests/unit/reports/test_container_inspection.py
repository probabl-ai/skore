import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from skrub import tabular_pipeline

from skore import CrossValidationReport, EstimatorReport, PermutationImportanceDisplay
from skore._externals._sklearn_compat import convert_container
from skore._sklearn._plot import TableReportDisplay


@pytest.mark.parametrize(
    "x_container,y_container",
    [
        ("array", "array"),
        ("pandas", "series"),
        ("polars", "polars_series"),
    ],
)
@pytest.mark.parametrize(
    "report_cls",
    [EstimatorReport, CrossValidationReport],
)
def test_permutation_importance_with_containers(report_cls, x_container, y_container):
    """Permutation importance accepts array, pandas, and polars X/y inputs."""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    feature_columns = [f"Feature {i}" for i in range(X.shape[1])]
    X = convert_container(
        X, x_container, column_names=feature_columns, minversion="0.20.23"
    )
    y = convert_container(y, y_container, minversion="0.20.23")
    estimator = Ridge()

    if report_cls is EstimatorReport:
        report = EstimatorReport(estimator, X_train=X, y_train=y, X_test=X, y_test=y)
    else:
        report = CrossValidationReport(estimator, X, y, splitter=2)

    display = report.inspection.permutation_importance(seed=42, n_repeats=2)
    assert isinstance(display, PermutationImportanceDisplay)


@pytest.mark.parametrize(
    "x_container,y_container",
    [
        ("array", "array"),
        ("pandas", "series"),
        ("polars", "polars_series"),
    ],
)
@pytest.mark.parametrize(
    "report_cls",
    [EstimatorReport, CrossValidationReport],
)
def test_data_summarize_plot_with_containers(
    report_cls, x_container, y_container, pyplot
):
    """Table report plots work with array, pandas, and polars-backed summaries."""
    X, y = make_regression(n_samples=100, n_features=3, random_state=42)
    feature_columns = [f"Feature {i}" for i in range(X.shape[1])]
    X = convert_container(
        X, x_container, column_names=feature_columns, minversion="0.20.23"
    )
    y = convert_container(y, y_container, minversion="0.20.23")
    estimator = tabular_pipeline(LinearRegression())

    if report_cls is EstimatorReport:
        report = EstimatorReport(estimator, X_train=X, y_train=y, X_test=X, y_test=y)
        display = report.data.summarize(data_source="train")
    else:
        report = CrossValidationReport(estimator, X, y, splitter=2)
        display = report.data.summarize()

    assert isinstance(display, TableReportDisplay)
    fig = display.plot(x=feature_columns[0], y=feature_columns[1])
    assert fig.axes
