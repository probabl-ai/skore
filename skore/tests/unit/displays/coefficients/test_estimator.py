import matplotlib as mpl
import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from skore import EstimatorReport


@pytest.mark.parametrize(
    "fixture_name, subplot_by, err_msg",
    [
        (
            "estimator_reports_binary_classification",
            "label",
            "No columns to group by.",
        ),
        (
            "estimator_reports_regression",
            "output",
            "No columns to group by.",
        ),
        (
            "estimator_reports_multiclass_classification",
            "incorrect",
            "Column incorrect not found in the frame. "
            + "It should be one of label, auto, None.",
        ),
        (
            "estimator_reports_multioutput_regression",
            "incorrect",
            "Column incorrect not found in the frame. "
            + "It should be one of output, auto, None.",
        ),
    ],
)
def test_invalid_subplot_by(fixture_name, subplot_by, err_msg, request):
    reports = request.getfixturevalue(fixture_name)
    report = reports[0]
    display = report.inspection.coefficients()
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by=subplot_by)


@pytest.mark.parametrize(
    "fixture_name, subplot_by_tuples",
    [
        (
            "estimator_reports_binary_classification",
            [(None, 0)],
        ),
        (
            "estimator_reports_multiclass_classification",
            [("label", 3), (None, 0)],
        ),
        (
            "estimator_reports_regression",
            [(None, 0)],
        ),
        (
            "estimator_reports_multioutput_regression",
            [("output", 2), (None, 0)],
        ),
    ],
)
def test_valid_subplot_by(fixture_name, subplot_by_tuples, request):
    """Check that we can pass non default values to `subplot_by`."""
    reports = request.getfixturevalue(fixture_name)
    report = reports[0]
    display = report.inspection.coefficients()
    for subplot_by, expected_len in subplot_by_tuples:
        fig = display.plot(subplot_by=subplot_by)
        axes = fig.axes
        if subplot_by is None:
            assert len(axes) == 1
            assert isinstance(axes[0], mpl.axes.Axes)
        else:
            assert len(axes) == expected_len


def test_include_intercept_multioutput_fit_intercept_false(request):
    """fit_intercept=False multi-output: scalar intercept is repeated per output."""
    X_train, X_test, y_train, y_test = request.getfixturevalue(
        "multioutput_regression_train_test_split"
    )
    report = EstimatorReport(
        LinearRegression(fit_intercept=False),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.inspection.coefficients()
    frame = display.frame(include_intercept=True)
    intercept_rows = frame.query("feature == 'Intercept'")
    assert len(intercept_rows) == 2
    assert set(intercept_rows["output"].astype(str)) == {"0", "1"}
    np.testing.assert_array_equal(intercept_rows["coefficient"].values, [0.0, 0.0])
    assert display.frame(include_intercept=False).query("feature == 'Intercept'").empty


def test_scale_features_multiplies_by_train_std(regression_train_test_split):
    """scale_features multiplies non-intercept coefficients by training feature std."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        Ridge(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.inspection.coefficients()
    assert display.coefficients["feature_std"].notna().all()

    raw = display.frame(include_intercept=False).set_index("feature")
    scaled = display.frame(include_intercept=False, scale_features=True).set_index(
        "feature"
    )
    std = (
        display.coefficients.query("feature != 'Intercept'")
        .set_index("feature")["feature_std"]
        .groupby(level=0)
        .first()
    )
    expected = raw["coefficient"] * std.loc[raw.index]
    np.testing.assert_allclose(scaled["coefficient"], expected)
    assert "feature_std" not in scaled.columns


def test_scale_features_leaves_intercept_unchanged(regression_train_test_split):
    """The intercept is not multiplied by a feature standard deviation."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        Ridge(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.inspection.coefficients()
    raw = display.frame().set_index("feature")
    scaled = display.frame(scale_features=True).set_index("feature")
    assert raw.loc["Intercept", "coefficient"] == scaled.loc["Intercept", "coefficient"]


def test_scale_features_uses_preprocessed_train_std(regression_train_test_split):
    """Feature stds are computed after the pipeline preprocessor."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.inspection.coefficients()
    # After StandardScaler (population std), sample stds are still ~1, and the
    # intercept is given a unit standard deviation.
    np.testing.assert_allclose(display.coefficients["feature_std"], 1.0, atol=0.05)
    raw = display.frame(include_intercept=False)["coefficient"].to_numpy()
    scaled = display.frame(include_intercept=False, scale_features=True)[
        "coefficient"
    ].to_numpy()
    np.testing.assert_allclose(raw, scaled, rtol=0.05)


def test_scale_features_sparse_preprocessor(regression_train_test_split):
    """Feature stds are computed on sparse transformed data without densifying."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        make_pipeline(SplineTransformer(sparse_output=True), Ridge()),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.inspection.coefficients()
    assert display.coefficients["feature_std"].notna().all()

    X_transformed = report.estimator_[:-1].transform(X_train)
    np.testing.assert_allclose(
        display.coefficients.query("feature != 'Intercept'")["feature_std"],
        np.std(X_transformed.toarray(), axis=0),
    )


def test_scale_features_prefit_without_train_data_raises():
    """Prefit reports without X_train cannot use scale_features."""
    X, y = make_regression(n_samples=50, n_features=4, random_state=0)
    estimator = Ridge().fit(X, y)
    report = EstimatorReport(estimator, X_test=X, y_test=y)
    display = report.inspection.coefficients()
    assert display.coefficients["feature_std"].isna().all()
    with pytest.raises(ValueError, match="training feature standard deviations"):
        display.frame(scale_features=True)
    with pytest.raises(ValueError, match="training feature standard deviations"):
        display.plot(scale_features=True)


def test_scale_features_plot_xlabel(regression_train_test_split):
    """Plot xlabel reflects scaled coefficients when requested."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        Ridge(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    fig = report.inspection.coefficients().plot(scale_features=True)
    assert fig.axes[0].get_xlabel() == "Magnitude of scaled coefficient"
