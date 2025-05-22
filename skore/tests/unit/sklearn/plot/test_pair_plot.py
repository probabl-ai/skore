import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore.sklearn._plot import PairPlotDisplay


@pytest.fixture
def binary_classification_data():
    X, y = make_classification(class_sep=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return (
        LogisticRegression().fit(X_train, y_train),
        X_train,
        X_test,
        y_train,
        y_test,
    )


@pytest.fixture
def multiclass_classification_data():
    X, y = make_classification(
        class_sep=0.1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return (
        LogisticRegression().fit(X_train, y_train),
        X_train,
        X_test,
        y_train,
        y_test,
    )


@pytest.fixture
def regression_data():
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return (
        LogisticRegression().fit(X_train, y_train),
        X_train,
        X_test,
        y_train,
        y_test,
    )


def test_pairplot_test_set(
    pyplot,
    binary_classification_data,
):
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.metrics.pairwise_plot(
        perf_metric_x="fit_time",
        perf_metric_y="predict_time",
        data_source="train",
    )

    display.plot()

    assert isinstance(display, PairPlotDisplay)
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")


def test_pairplot_train_set(
    pyplot,
    binary_classification_data,
):
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    display = report.metrics.pairwise_plot(
        perf_metric_x="fit_time",
        perf_metric_y="roc_auc",
        data_source="train",
        pos_label=1,
    )

    display.plot()

    assert isinstance(display, PairPlotDisplay)
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")


def test_pairplot_missing_col(
    pyplot,
    binary_classification_data,
):
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    with pytest.raises(ValueError):
        report.metrics.pairwise_plot(
            perf_metric_x="fit_time",
            perf_metric_y="an_invented_column",
            data_source="train",
            pos_label=1,
        )


def test_pairplot_missing_pos_label(
    pyplot,
    binary_classification_data,
):
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    with pytest.raises(ValueError):
        report.metrics.pairwise_plot(
            perf_metric_x="fit_time",
            perf_metric_y="recall",
            data_source="train",
        )
