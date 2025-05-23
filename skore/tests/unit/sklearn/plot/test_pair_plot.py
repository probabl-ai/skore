import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from skore import ComparisonReport, EstimatorReport
from skore.sklearn._plot import PairPlotDisplay


@pytest.fixture
def binary_classification_data():
    X, y = make_classification(class_sep=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def test_pairplot_test_set(
    pyplot,
    binary_classification_data,
):
    X_train, X_test, y_train, y_test = binary_classification_data
    est_1 = LogisticRegression()
    est_2 = DecisionTreeClassifier()
    report1 = EstimatorReport(
        est_1,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report2 = EstimatorReport(
        est_2,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    comparison = ComparisonReport(
        {"Logistic Regression": report1, "Decision Tree": report2}
    )
    display = comparison.pairwise_plot(
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
    X_train, X_test, y_train, y_test = binary_classification_data
    est_1 = LogisticRegression()
    est_2 = DecisionTreeClassifier()
    report1 = EstimatorReport(
        est_1,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report2 = EstimatorReport(
        est_2,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    comparison = ComparisonReport(
        {"Logistic Regression": report1, "Decision Tree": report2}
    )

    display = comparison.pairwise_plot(
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
    X_train, X_test, y_train, y_test = binary_classification_data
    est_1 = LogisticRegression()
    est_2 = DecisionTreeClassifier()
    report1 = EstimatorReport(
        est_1,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report2 = EstimatorReport(
        est_2,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    comparison = ComparisonReport(
        {"Logistic Regression": report1, "Decision Tree": report2}
    )

    with pytest.raises(ValueError):
        comparison.pairwise_plot(
            perf_metric_x="fit_time",
            perf_metric_y="an_invented_column",
            data_source="train",
            pos_label=1,
        )


def test_pairplot_missing_pos_label(
    pyplot,
    binary_classification_data,
):
    X_train, X_test, y_train, y_test = binary_classification_data
    est_1 = LogisticRegression()
    est_2 = DecisionTreeClassifier()
    report1 = EstimatorReport(
        est_1,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report2 = EstimatorReport(
        est_2,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    comparison = ComparisonReport(
        {"Logistic Regression": report1, "Decision Tree": report2}
    )

    with pytest.raises(ValueError):
        comparison.pairwise_plot(
            perf_metric_x="fit_time",
            perf_metric_y="recall",
            data_source="train",
        )
