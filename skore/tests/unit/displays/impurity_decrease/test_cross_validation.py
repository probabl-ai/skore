import matplotlib as mpl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import _convert_container

from skore import CrossValidationReport, ImpurityDecreaseDisplay


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
    estimator = clone(estimator)

    columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
    X = _convert_container(X, "dataframe", columns_name=columns_names)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("predictor", estimator),
        ]
    )
    report = CrossValidationReport(model, X, y, splitter=2)
    display = report.inspection.impurity_decrease()
    assert isinstance(display, ImpurityDecreaseDisplay)
    assert list(display.importances.columns) == [
        "estimator",
        "split",
        "feature",
        "importance",
    ]
    assert display.importances["split"].nunique() == 2
    for split_index, est_report in enumerate(report.estimator_reports_):
        fitted = est_report.estimator_.named_steps["predictor"]
        imp = (
            display.importances.query(f"split == {split_index}")
            .set_index("feature")
            .loc[columns_names, "importance"]
            .to_numpy()
        )
        np.testing.assert_allclose(imp, fitted.feature_importances_)
    frame = display.frame()
    assert list(frame.columns) == ["split", "feature", "importance"]
    display.plot()
    assert hasattr(display, "facet_") and hasattr(display, "figure_")
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert display.ax_.get_xlabel() == "Mean decrease in impurity"
