import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils._testing import _convert_container

from skore import CoefficientsDisplay, ComparisonReport, EstimatorReport


def test_invalid_subplot_by(
    pyplot, coefficients_comparison_estimator_reports_binary_classification
):
    display = coefficients_comparison_estimator_reports_binary_classification.inspection.coefficients()
    with pytest.raises(ValueError, match="Column incorrect not found in the frame"):
        display.plot(subplot_by="incorrect")


def test_subplot_by_estimator(
    pyplot, coefficients_comparison_estimator_reports_binary_classification
):
    report = coefficients_comparison_estimator_reports_binary_classification
    display = report.inspection.coefficients()
    display.plot(subplot_by="estimator")
    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == len(report.reports_)
    for report_name, ax in zip(report.reports_, display.ax_, strict=True):
        assert isinstance(ax, mpl.axes.Axes)
        assert ax.get_title() == f"estimator = {report_name}"


@pytest.mark.parametrize(
    "fixture_name",
    [
        "logistic_multiclass_classification_with_train_test",
        "linear_regression_multioutput_with_train_test",
    ],
)
def test_subplot_by_none_multiclass_or_multioutput(
    pyplot,
    request,
    fixture_name,
):
    """Check that an error is raised when `subplot_by=None` and there are multiple
    labels (multiclass) or outputs (multi-output regression)."""
    fixture = request.getfixturevalue(fixture_name)
    estimator, X_train, X_test, y_train, y_test = fixture
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    report_1 = EstimatorReport(
        clone(estimator), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        clone(estimator), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

    display = report.inspection.coefficients()

    err_msg = (
        "There are multiple labels or outputs and `subplot_by` is `None`. "
        "There is too much information to display on a single plot. "
        "Please provide a column to group by using `subplot_by`."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by=None)


def test_different_features(
    pyplot,
    logistic_multiclass_classification_with_train_test,
):
    """Check that we get a proper report even if the estimators do not have the same
    input features."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)
    n_classes = len(np.unique(y_train))

    simple_model = clone(estimator)

    preprocessor = ColumnTransformer(
        [("poly", PolynomialFeatures(), columns_names[:2])],
        remainder="passthrough",
    )

    complex_model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("predictor", clone(estimator)),
        ]
    )

    report_simple = EstimatorReport(
        simple_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_complex = EstimatorReport(
        complex_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"report_simple": report_simple, "report_complex": report_complex}
    )

    display = report.inspection.coefficients()
    assert isinstance(display, CoefficientsDisplay)

    df = display.frame(sorting_order=None)
    expected_features = [
        "Intercept"
    ] + report_simple.estimator_.feature_names_in_.tolist()
    assert (
        df.query("estimator == 'report_simple'")["feature"].tolist()
        == expected_features * n_classes
    )

    expected_features = ["Intercept"] + report_complex.estimator_[
        :-1
    ].get_feature_names_out().tolist()
    assert (
        df.query("estimator == 'report_complex'")["feature"].tolist()
        == expected_features * n_classes
    )

    err_msg = (
        "The estimators have different features and should be plotted on different "
        "axis using `subplot_by='estimator'`."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="label")

    for subplot_by in ("auto", "estimator"):
        display.plot(subplot_by=subplot_by)
        assert hasattr(display, "facet_")
        assert hasattr(display, "figure_")
        assert hasattr(display, "ax_")
        assert isinstance(display.ax_, np.ndarray)
        assert len(display.ax_) == len(report.reports_)
        for report_name, ax in zip(report.reports_, display.ax_, strict=True):
            assert isinstance(ax, mpl.axes.Axes)
            assert ax.get_title() == f"estimator = {report_name}"
            assert ax.get_xlabel() == "Magnitude of coefficient"
            assert ax.get_ylabel() == ""
        assert display.figure_.get_suptitle() == "Coefficients by estimator"


def test_include_intercept(
    pyplot,
    logistic_binary_classification_with_train_test,
):
    """Check whether or not we can include or exclude the intercept."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)

    report_1 = EstimatorReport(
        clone(estimator), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        clone(estimator), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

    display = report.inspection.coefficients()

    assert display.frame(include_intercept=False).query("feature == 'Intercept'").empty

    display.plot(include_intercept=False)
    assert all(
        label.get_text() != "Intercept" for label in display.ax_.get_yticklabels()
    )
    assert display.figure_.get_suptitle() == "Coefficients"
