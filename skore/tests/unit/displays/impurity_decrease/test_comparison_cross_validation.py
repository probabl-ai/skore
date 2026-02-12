import matplotlib as mpl
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import _convert_container

from skore import ComparisonReport, CrossValidationReport, ImpurityDecreaseDisplay


def test_with_pipeline(pyplot, forest_binary_classification_data):
    estimator, X, y = forest_binary_classification_data
    columns_names = [f"Feature #{i}" for i in range(X.shape[1])]
    X = _convert_container(X, "dataframe", columns_name=columns_names)
    splitter = 2

    predictor = clone(estimator)
    model = Pipeline([("scaler", StandardScaler()), ("predictor", predictor)])

    report_1 = CrossValidationReport(model, X, y, splitter=splitter)
    report_2 = CrossValidationReport(model, X, y, splitter=splitter)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

    display = report.inspection.impurity_decrease()
    assert isinstance(display, ImpurityDecreaseDisplay)

    expected_columns = ["estimator", "split", "feature", "importance"]
    df = display.importances
    assert sorted(df.columns.tolist()) == sorted(expected_columns)
    assert df["split"].nunique() == splitter
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())

    for report_name, cv_report in report.reports_.items():
        for split_index, estimator_report in enumerate(cv_report.estimator_reports_):
            fitted_predictor = estimator_report.estimator_.named_steps["predictor"]
            feature_importances = fitted_predictor.feature_importances_

            importances_split = (
                df.query(f"estimator == '{report_name}' & split == {split_index}")
                .loc[:, ["feature", "importance"]]
                .set_index("feature")
                .loc[columns_names]
                .to_numpy()
                .flatten()
            )

            np.testing.assert_allclose(importances_split, feature_importances)

    df = display.frame()
    expected_columns = ["estimator", "split", "feature", "importance"]
    assert df.columns.tolist() == expected_columns
    assert df["feature"].tolist() == columns_names * splitter * len(report.reports_)
    assert df["estimator"].unique().tolist() == list(report.reports_.keys())
    assert df["split"].nunique() == splitter

    display.plot()
    assert hasattr(display, "facet_")
    assert hasattr(display, "figure_")
    assert hasattr(display, "ax_")
    assert isinstance(display.ax_, mpl.axes.Axes)

    assert display.figure_.get_suptitle() == "Mean decrease in impurity (MDI)"
    assert display.ax_.get_xlabel() == "Mean decrease in impurity"
    yticklabels = [label.get_text() for label in display.ax_.get_yticklabels()]
    assert yticklabels == ["Feature #0", "Feature #1", "Feature #2", "Feature #3"]
