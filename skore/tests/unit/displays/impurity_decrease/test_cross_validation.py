import matplotlib as mpl
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import _convert_container

from skore import CrossValidationReport, ImpurityDecreaseDisplay


def test_with_pipeline(pyplot, forest_binary_classification_data):
    estimator, X, y = forest_binary_classification_data
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
    for split_index, est_report in enumerate(report.reports_):
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
    assert display.ax_.get_xlabel() == "Mean Decrease in Impurity (MDI)"
