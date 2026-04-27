import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

from skore import ComparisonReport, CrossValidationReport, EstimatorReport, evaluate


def test_prefit_estimator(regression_data):
    """A fitted estimator with splitter='prefit' returns an EstimatorReport."""
    X, y = regression_data
    fitted = LinearRegression().fit(X, y)
    report = evaluate(fitted, X, y, splitter="prefit")
    assert isinstance(report, EstimatorReport)
    assert report.X_train is None
    assert report.y_train is None
    assert report.X_test is X
    assert report.y_test is y


def test_prefit_unfitted_raises(regression_data):
    """An unfitted estimator with splitter='prefit' raises an error."""
    X, y = regression_data
    with pytest.raises(ValueError):
        evaluate(LinearRegression(), X, y, splitter="prefit")


def test_float_splitter(regression_data):
    """A float splitter triggers a single split and returns an EstimatorReport."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=0.3)
    assert isinstance(report, EstimatorReport)
    assert len(report.X_test) == 0.3 * len(X)


def test_default_splitter(regression_data):
    """Calling without splitter uses the default and returns an EstimatorReport."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y)
    assert isinstance(report, EstimatorReport)
    assert len(report.X_test) == 0.2 * len(X)


def test_int_splitter(regression_data):
    """An int splitter triggers cross-validation."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=3)
    assert isinstance(report, CrossValidationReport)
    assert len(report.estimator_reports_) == 3


def test_cv_object_splitter(regression_data):
    """A CV object splitter triggers cross-validation."""
    X, y = regression_data
    report = evaluate(LinearRegression(), X, y, splitter=KFold(n_splits=4))
    assert isinstance(report, CrossValidationReport)
    assert len(report.estimator_reports_) == 4


def test_multiple_estimators(regression_data):
    """A list of estimators returns a ComparisonReport."""
    X, y = regression_data
    estimators = [LinearRegression(), LinearRegression()]
    report = evaluate(estimators, X, y, splitter=0.2)
    assert isinstance(report, ComparisonReport)


def test_multiple_estimators_prefit(regression_data):
    """A list of fitted estimators with splitter='prefit' returns ComparisonReport."""
    X, y = regression_data
    fitted1 = LinearRegression().fit(X, y)
    fitted2 = LinearRegression().fit(X, y)
    report = evaluate([fitted1, fitted2], X, y, splitter="prefit")
    assert isinstance(report, ComparisonReport)


def test_multiple_estimators_cv(regression_data):
    """A list of estimators with int splitter returns a ComparisonReport."""
    X, y = regression_data
    report = evaluate([LinearRegression(), LinearRegression()], X, y, splitter=3)
    assert isinstance(report, ComparisonReport)


def test_multiple_estimators_multiple_X(regression_data):
    """A list of estimators with a list of X returns a ComparisonReport."""
    X, y = regression_data
    report = evaluate([LinearRegression(), LinearRegression()], [X, X], y, splitter=0.2)
    assert isinstance(report, ComparisonReport)


def test_list_estimator_dict_X_raises(regression_data):
    """List estimators with a dict X raises TypeError."""
    X, y = regression_data
    with pytest.raises(TypeError, match="cannot be a dict"):
        evaluate(
            [LinearRegression(), LinearRegression()],
            {"a": X, "b": X},
            y,
            splitter=0.2,
        )


def test_multiple_estimators_dict(regression_data):
    """A dict of named estimators returns a ComparisonReport with matching keys."""
    X, y = regression_data
    report = evaluate(
        {"a": LinearRegression(), "b": LinearRegression()},
        X,
        y,
        splitter=0.2,
    )
    assert isinstance(report, ComparisonReport)
    assert set(report.reports_) == {"a", "b"}


def test_multiple_estimators_dict_cv(regression_data):
    """A dict of estimators with int splitter returns a ComparisonReport of CV
    reports."""
    X, y = regression_data
    report = evaluate(
        {"a": LinearRegression(), "b": LinearRegression()},
        X,
        y,
        splitter=3,
    )
    assert isinstance(report, ComparisonReport)
    assert all(isinstance(r, CrossValidationReport) for r in report.reports_.values())
    assert set(report.reports_) == {"a", "b"}


def test_multiple_estimators_dict_per_estimator_X(regression_data):
    """A dict of estimators with a dict of X (same keys) returns a ComparisonReport."""
    X, y = regression_data
    report = evaluate(
        {"a": LinearRegression(), "b": LinearRegression()},
        {"a": X, "b": X},
        y,
        splitter=0.2,
    )
    assert isinstance(report, ComparisonReport)
    assert set(report.reports_) == {"a", "b"}


def test_dict_estimator_list_X_raises(regression_data):
    """Dict estimators with a list X raises TypeError."""
    X, y = regression_data
    with pytest.raises(TypeError, match="X cannot be a list"):
        evaluate(
            {"a": LinearRegression(), "b": LinearRegression()},
            [X, X],
            y,
            splitter=0.2,
        )


def test_dict_estimator_mismatched_X_keys_raises(regression_data):
    """Dict estimators with dict X whose keys differ from estimator raises
    ValueError."""
    X, y = regression_data
    with pytest.raises(ValueError, match="same keys"):
        evaluate(
            {"a": LinearRegression(), "b": LinearRegression()},
            {"a": X, "c": X},
            y,
            splitter=0.2,
        )


def test_dict_estimators_prefit(regression_data):
    """A dict of fitted estimators with splitter='prefit' returns ComparisonReport."""
    X, y = regression_data
    fitted1 = LinearRegression().fit(X, y)
    fitted2 = LinearRegression().fit(X, y)
    report = evaluate({"a": fitted1, "b": fitted2}, X, y, splitter="prefit")
    assert isinstance(report, ComparisonReport)
    assert set(report.reports_) == {"a", "b"}


def test_dict_estimators_prefit_X_none(regression_data):
    """A dict of prefit estimators with X=None returns ComparisonReport."""
    X, y = regression_data
    fitted1 = LinearRegression().fit(X, y)
    fitted2 = LinearRegression().fit(X, y)
    report = evaluate({"a": fitted1, "b": fitted2}, None, y, splitter="prefit")
    assert isinstance(report, ComparisonReport)
    assert set(report.reports_) == {"a", "b"}


def test_empty_dict_raises():
    """An empty estimator dict raises ValueError."""
    with pytest.raises(ValueError, match="Expected.*reports to compare"):
        evaluate({}, None, None)


def test_single_estimator_list_X_raises(regression_data):
    """A single estimator with list X raises TypeError."""
    X, y = regression_data
    with pytest.raises(TypeError, match="single array-like"):
        evaluate(LinearRegression(), [X], y)


def test_single_estimator_dict_X_raises(regression_data):
    """A single estimator with dict X raises TypeError."""
    X, y = regression_data
    with pytest.raises(TypeError, match="single array-like"):
        evaluate(LinearRegression(), {"a": X}, y)


def test_invalid_splitter_string(regression_data):
    """An invalid string splitter raises ValueError."""
    X, y = regression_data
    with pytest.raises(ValueError, match="Invalid string value for splitter"):
        evaluate(LinearRegression(), X, y, splitter="invalid")


def test_classification(binary_classification_data):
    """evaluate works with a classification estimator."""
    X, y = binary_classification_data
    report = evaluate(LogisticRegression(), X, y, splitter=0.2)
    assert isinstance(report, EstimatorReport)


def test_classification_cv(binary_classification_data):
    """evaluate works with classification + CV splitter object."""
    X, y = binary_classification_data
    report = evaluate(LogisticRegression(), X, y, splitter=StratifiedKFold(n_splits=3))
    assert isinstance(report, CrossValidationReport)
    assert len(report.estimator_reports_) == 3


def test_pos_label(binary_classification_data):
    """pos_label is forwarded to the underlying report."""
    X, y = binary_classification_data
    report = evaluate(LogisticRegression(), X, y, splitter=0.2, pos_label=0)
    assert isinstance(report, EstimatorReport)
    assert report.pos_label == 0
