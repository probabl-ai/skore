import matplotlib as mpl
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from skore import ComparisonReport, CrossValidationReport, EstimatorReport

mpl.rc("figure", max_open_warning=False)


@pytest.fixture(scope="module")
def estimator_reports_binary_classification_figure_axes(
    pyplot, estimator_reports_binary_classification
):
    report = estimator_reports_binary_classification[0]
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def estimator_reports_multiclass_classification_figure_axes(
    pyplot, estimator_reports_multiclass_classification
):
    report = estimator_reports_multiclass_classification[0]
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def estimator_reports_regression_figure_axes(pyplot, estimator_reports_regression):
    report = estimator_reports_regression[0]
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def estimator_reports_multioutput_regression_figure_axes(
    pyplot, estimator_reports_multioutput_regression
):
    report = estimator_reports_multioutput_regression[0]
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def cross_validation_reports_binary_classification_figure_axes(
    pyplot, cross_validation_reports_binary_classification
):
    report = cross_validation_reports_binary_classification[0]
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def cross_validation_reports_multiclass_classification_figure_axes(
    pyplot, cross_validation_reports_multiclass_classification
):
    report = cross_validation_reports_multiclass_classification[0]
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def cross_validation_reports_regression_figure_axes(
    pyplot, cross_validation_reports_regression
):
    report = cross_validation_reports_regression[0]
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def cross_validation_reports_multioutput_regression_figure_axes(
    pyplot, cross_validation_reports_multioutput_regression
):
    report = cross_validation_reports_multioutput_regression[0]
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def comparison_estimator_reports_binary_classification_figure_axes(
    pyplot, comparison_estimator_reports_binary_classification
):
    report = comparison_estimator_reports_binary_classification
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def comparison_estimator_reports_multiclass_classification_figure_axes(
    pyplot, comparison_estimator_reports_multiclass_classification
):
    report = comparison_estimator_reports_multiclass_classification
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def comparison_estimator_reports_regression_figure_axes(
    pyplot, comparison_estimator_reports_regression
):
    report = comparison_estimator_reports_regression
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def comparison_estimator_reports_multioutput_regression_figure_axes(
    pyplot, comparison_estimator_reports_multioutput_regression
):
    report = comparison_estimator_reports_multioutput_regression
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_binary_classification_figure_axes(
    pyplot,
    comparison_cross_validation_reports_binary_classification,
):
    report = comparison_cross_validation_reports_binary_classification
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_multiclass_classification_figure_axes(
    pyplot,
    comparison_cross_validation_reports_multiclass_classification,
):
    report = comparison_cross_validation_reports_multiclass_classification
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_regression_figure_axes(
    pyplot,
    comparison_cross_validation_reports_regression,
):
    report = comparison_cross_validation_reports_regression
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_multioutput_regression_figure_axes(
    pyplot,
    comparison_cross_validation_reports_multioutput_regression,
):
    report = comparison_cross_validation_reports_multioutput_regression
    display = report.inspection.coefficients()
    display.plot()
    return display.figure_, display.ax_


@pytest.fixture(scope="module")
def binary_classification_half(binary_classification):
    X, y = binary_classification
    X = X[:, :2]
    return X, y


@pytest.fixture(scope="module")
def binary_classification_train_test_split_half(binary_classification_half):
    X, y = binary_classification_half
    return train_test_split(X, y, test_size=0.2, random_state=0)


@pytest.fixture(scope="module")
def estimator_report_binary_classification_half(
    binary_classification_train_test_split_half,
):
    X_train, X_test, y_train, y_test = binary_classification_train_test_split_half
    return EstimatorReport(
        LogisticRegression(random_state=0),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@pytest.fixture(scope="module")
def cross_validation_report_binary_classification_half(
    binary_classification_half,
):
    X, y = binary_classification_half
    return CrossValidationReport(
        LogisticRegression(random_state=0), X=X, y=y, splitter=2
    )


@pytest.fixture(scope="module")
def comparison_estimator_reports_binary_classification_different_features(
    estimator_reports_binary_classification,
    estimator_report_binary_classification_half,
):
    report_1 = estimator_reports_binary_classification[0]
    report_2 = estimator_report_binary_classification_half
    return ComparisonReport([report_1, report_2])


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_binary_classification_different_features(
    cross_validation_reports_binary_classification,
    cross_validation_report_binary_classification_half,
):
    report_1 = cross_validation_reports_binary_classification[0]
    report_2 = cross_validation_report_binary_classification_half
    return ComparisonReport([report_1, report_2])


@pytest.fixture(scope="module")
def multiclass_classification_half(multiclass_classification):
    X, y = multiclass_classification
    X = X[:, :2]
    return X, y


@pytest.fixture(scope="module")
def multiclass_classification_train_test_split_half(multiclass_classification_half):
    X, y = multiclass_classification_half
    return train_test_split(X, y, test_size=0.2, random_state=0)


@pytest.fixture(scope="module")
def estimator_report_multiclass_classification_half(
    multiclass_classification_train_test_split_half,
):
    X_train, X_test, y_train, y_test = multiclass_classification_train_test_split_half
    return EstimatorReport(
        LogisticRegression(random_state=0),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@pytest.fixture(scope="module")
def cross_validation_report_multiclass_classification_half(
    multiclass_classification_half,
):
    X, y = multiclass_classification_half
    return CrossValidationReport(
        LogisticRegression(random_state=0), X=X, y=y, splitter=2
    )


@pytest.fixture(scope="module")
def comparison_estimator_reports_multiclass_classification_different_features(
    estimator_reports_multiclass_classification,
    estimator_report_multiclass_classification_half,
):
    report_1 = estimator_reports_multiclass_classification[0]
    report_2 = estimator_report_multiclass_classification_half
    return ComparisonReport([report_1, report_2])


@pytest.fixture(scope="module")
def comparison_cross_validation_reports_multiclass_classification_different_features(
    cross_validation_reports_multiclass_classification,
    cross_validation_report_multiclass_classification_half,
):
    report_1 = cross_validation_reports_multiclass_classification[0]
    report_2 = cross_validation_report_multiclass_classification_half
    return ComparisonReport([report_1, report_2])
