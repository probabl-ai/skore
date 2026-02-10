import matplotlib as mpl
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import _convert_container

from skore import EstimatorReport, ImpurityDecreaseDisplay


def test_with_pipeline(pyplot, forest_binary_classification_with_train_test):
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = _convert_container(X_train, "dataframe", columns_name=columns_names)
    X_test = _convert_container(X_test, "dataframe", columns_name=columns_names)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("predictor", estimator),
        ]
    )
    report = EstimatorReport(
        model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.impurity_decrease()
    assert isinstance(display, ImpurityDecreaseDisplay)
    assert set(display.importances.columns) == {
        "estimator",
        "split",
        "feature",
        "importance",
    }
    fitted_predictor = report.estimator_.named_steps["predictor"]
    np.testing.assert_allclose(
        display.importances["importance"].to_numpy(),
        fitted_predictor.feature_importances_,
    )
    frame = display.frame()
    assert list(frame.columns) == ["feature", "importance"]
    assert frame["feature"].tolist() == columns_names
    display.plot()
    assert hasattr(display, "figure_") and hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert display.ax_.get_xlabel() == "Mean Decrease in Impurity (MDI)"
