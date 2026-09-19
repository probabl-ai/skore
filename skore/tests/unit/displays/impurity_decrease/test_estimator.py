import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
import skrub
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from skore import EstimatorReport, ImpurityDecreaseDisplay
from skore._externals.sklearn_compat import convert_container


def test_with_pipeline(forest_binary_classification_with_train_test):
    estimator, X_train, X_test, y_train, y_test = (
        forest_binary_classification_with_train_test
    )
    estimator = clone(estimator)
    columns_names = [f"Feature #{i}" for i in range(X_train.shape[1])]
    X_train = convert_container(X_train, "pandas", column_names=columns_names)
    X_test = convert_container(X_test, "pandas", column_names=columns_names)
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
    fig = display.plot()
    ax = fig.axes[0]
    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)
    estimator_name = display.importances["estimator"][0]
    assert fig.get_suptitle() == f"Mean decrease in impurity (MDI) of {estimator_name}"
    assert ax.get_xlabel() == "Mean decrease in impurity"
    yticklabels = [label.get_text() for label in ax.get_yticklabels()]
    assert yticklabels == ["Feature #0", "Feature #1", "Feature #2", "Feature #3"]


@pytest.mark.parametrize(
    "data_op",
    [
        pytest.param(
            lambda forest: (
                skrub.X().skb.apply(StandardScaler()).skb.apply(forest, y=skrub.y())
            ),
            id="chained_applies",
        ),
        pytest.param(
            lambda forest: skrub.X().skb.apply(
                make_pipeline(StandardScaler(), forest), y=skrub.y()
            ),
            id="pipeline_in_apply",
        ),
    ],
)
def test_skrub_learner_matches_sklearn_pipeline(data_op):
    """A skrub learner gives the same importances as the equivalent scikit-learn
    pipeline."""
    X, y = make_classification(
        n_samples=60, n_features=3, n_redundant=0, random_state=0
    )
    X = pd.DataFrame(X, columns=["a", "b", "c"])
    forest = RandomForestClassifier(n_estimators=5, random_state=0)
    skrub_report = EstimatorReport(
        data_op(clone(forest)).skb.make_learner(),
        train_data={"X": X[:40], "y": y[:40]},
        test_data={"X": X[40:], "y": y[40:]},
    )
    sklearn_report = EstimatorReport(
        make_pipeline(StandardScaler(), clone(forest)),
        X_train=X[:40],
        y_train=y[:40],
        X_test=X[40:],
        y_test=y[40:],
    )
    pd.testing.assert_frame_equal(
        skrub_report.inspection.impurity_decrease().frame(),
        sklearn_report.inspection.impurity_decrease().frame(),
    )
