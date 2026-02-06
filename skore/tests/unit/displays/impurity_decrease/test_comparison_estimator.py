import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import _convert_container

from skore import ComparisonReport, EstimatorReport, ImpurityDecreaseDisplay


@pytest.mark.parametrize(
    "data_fixture",
    [
        "forest_binary_classification_data",
        "forest_multiclass_classification_data",
        "forest_regression_data",
        "forest_regression_multioutput_data",
    ],
)
@pytest.mark.parametrize("with_preprocessing", [True, False])
def test_impurity_decrease_comparison_estimator(
    pyplot,
    data_fixture,
    with_preprocessing,
    request,
):
    """Check the attributes and default plotting behaviour of the impurity decrease plot
    with comparison of estimator reports."""
    estimator, X, y = request.getfixturevalue(data_fixture)
    predictor = clone(estimator)

    columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
    X = _convert_container(X, "dataframe", columns_name=columns_names)

    if with_preprocessing:
        model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])
    else:
        model = predictor

    X_train, X_test = X[:75], X[75:]
    y_train, y_test = y[:75], y[75:]

    report_1 = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

    display = report.inspection.impurity_decrease()
    assert isinstance(display, ImpurityDecreaseDisplay)

    expected_columns = ["estimator", "split", "feature", "importance"]
    df = display.importances
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    assert df["split"].isna().all()
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())

    for report_name, estimator_report in report.reports_.items():
        fitted_predictor = estimator_report.estimator_
        if with_preprocessing:
            fitted_predictor = fitted_predictor.named_steps["predictor"]
        feature_importances = fitted_predictor.feature_importances_

        importances_split = (
            df.query(f"estimator == '{report_name}'")[["feature", "importance"]]
            .set_index("feature")
            .loc[columns_names]
            .to_numpy()
            .flatten()
        )

        np.testing.assert_allclose(importances_split, feature_importances)

    df = display.frame()
    expected_columns = ["estimator", "feature", "importance"]
    assert df.columns.tolist() == expected_columns
    assert df["feature"].tolist() == columns_names * len(report.reports_)
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())

    display.plot()
    assert hasattr(display, "facet_")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_xlabel() == "Mean decrease in impurity"
    assert display.ax_.get_ylabel() == ""
    assert display.figure_.get_suptitle() == "Mean decrease in impurity (MDI)"
