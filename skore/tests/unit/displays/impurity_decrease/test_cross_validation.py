import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import _convert_container

from skore import CrossValidationReport, ImpurityDecreaseDisplay


def _check_impurity_decrease_display(
    display, report, columns_names, splitter, with_preprocessing
):
    """Helper function to check impurity decrease display attributes."""
    assert isinstance(display, ImpurityDecreaseDisplay)

    expected_columns = ["estimator", "split", "feature", "importances"]
    df = display.importances
    assert df.columns.tolist() == expected_columns
    assert df["split"].nunique() == splitter
    assert df["estimator"].nunique() == 1

    for split_index, estimator_report in enumerate(report.estimator_reports_):
        fitted_predictor = estimator_report.estimator_
        if with_preprocessing:
            fitted_predictor = fitted_predictor.named_steps["predictor"]

        importances_split = (
            df.query(f"split == {split_index}")[["feature", "importances"]]
            .set_index("feature")
            .loc[columns_names]
            .to_numpy()
            .flatten()
        )

        np.testing.assert_allclose(
            importances_split, fitted_predictor.feature_importances_
        )

    df = display.frame()
    expected_columns = ["split", "feature", "importances"]
    assert df.columns.tolist() == expected_columns
    assert df["feature"].tolist() == columns_names * splitter
    assert df["split"].nunique() == splitter

    display.plot()
    assert hasattr(display, "facet_")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.ax_.get_xlabel() == "Mean decrease impurity"
    assert display.ax_.get_ylabel() == ""
    estimator_name = display.importances["estimator"][0]
    assert (
        display.figure_.get_suptitle() == f"Mean decrease impurity of {estimator_name}"
    )


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
def test_impurity_decrease_cross_validation(
    pyplot,
    data_fixture,
    with_preprocessing,
    request,
):
    """Check the attributes and default plotting behaviour of the impurity decrease plot
    with cross-validation reports."""
    estimator, X, y = request.getfixturevalue(data_fixture)
    predictor = clone(estimator)

    columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
    X = _convert_container(X, "dataframe", columns_name=columns_names)
    splitter = 2

    if with_preprocessing:
        model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])
    else:
        model = predictor

    report = CrossValidationReport(model, X, y, splitter=splitter)
    display = report.inspection.impurity_decrease()

    _check_impurity_decrease_display(
        display, report, columns_names, splitter, with_preprocessing
    )
