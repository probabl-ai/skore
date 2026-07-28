import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skore import evaluate
from skore._sklearn._checks._utils import CheckNotApplicable
from skore._sklearn._checks.model_checks import CheckHyperparamsAtSearchEdge
from skore._utils._testing import MockEstimator


@pytest.mark.parametrize(
    "estimator, param_name, side, expected",
    [
        # Ridge.alpha: Interval(Real, 0, None, closed='left') -> left bound is 0.0
        (Ridge(), "alpha", "left", 0.0),
        # Ridge.alpha has no finite right bound
        (Ridge(), "alpha", "right", None),
        # BayesianRidge.tol: Interval(Real, 0, None, closed='neither') -> open, no bound
        (BayesianRidge(), "tol", "left", None),
        # estimator without _parameter_constraints
        (MockEstimator(error=ValueError("unused")), "alpha", "left", None),
        # Pipeline: navigate 'ridge__alpha' to Ridge.alpha left bound
        (
            Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
            "ridge__alpha",
            "left",
            0.0,
        ),
    ],
)
def test_get_space_bound(estimator, param_name, side, expected):
    actual = CheckHyperparamsAtSearchEdge._get_space_bound(
        estimator, param_name=param_name, side=side
    )
    assert actual == expected


def _prefit_grid_search_report(X, y, search):
    search.fit(X, y)
    return evaluate(search, X, y, splitter="prefit")


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_raises_at_numeric_edge(report_type, regression_data):
    """SKD014 flags when best is at the numeric min of the searched values."""
    X, y = regression_data
    search = GridSearchCV(
        DummyRegressor(strategy="constant"),
        param_grid={"constant": [10.0, 0.1, 1.0]},
        cv=2,
    )

    if report_type == "estimator":
        report = evaluate(search, X, y)
        report.estimator_.best_params_ = {"constant": 0.1}
    else:
        report = evaluate(search, X, y, splitter=3)
        report.reports_[0].estimator_.best_params_ = {"constant": 0.1}

    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD014" in issues.index
    explanation = issues.loc["SKD014", "explanation"]
    assert "constant" in explanation
    assert "minimum" in explanation


def test_not_raised_for_interior_best(regression_data):
    """SKD014 is absent when the best value is not at the tried min or max."""
    X, y = regression_data
    search = GridSearchCV(
        DummyRegressor(strategy="constant"),
        param_grid={"constant": [3.0, 1.0, 2.0]},
        cv=2,
    )
    report = evaluate(search, X, y)
    report.estimator_.best_params_ = {"constant": 2.0}
    codes = set(report.checks.summarize().frame(section="issue")["code"])
    assert "SKD014" not in codes


def test_prefit(regression_data):
    """SKD014 runs for pre-fitted GridSearchCV reports."""
    X, y = regression_data
    search = GridSearchCV(
        DummyRegressor(strategy="constant"),
        param_grid={"constant": [10.0, 0.1, 1.0]},
        cv=2,
    )
    report = _prefit_grid_search_report(X, y, search)
    report.estimator_.best_params_ = {"constant": 0.1}
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD014" in issues.index
    assert "minimum" in issues.loc["SKD014", "explanation"]


@pytest.mark.parametrize(
    "param_grid", [{"fit_intercept": [False, True]}, {"solver": ["svd", "cholesky"]}]
)
def test_skips_non_numeric_hyperparameters(regression_data, param_grid):
    """SKD014 ignores bool, string, and other non-numeric search parameters."""
    X, y = regression_data
    search = GridSearchCV(Ridge(), param_grid=param_grid, cv=2)
    report = evaluate(search, X, y)
    codes = set(report.checks.summarize().frame(section="issue")["code"])
    assert "SKD014" not in codes


def test_skips_non_numeric_best_value(regression_data):
    """A non-numeric best value is skipped even though tried values are numeric."""
    X, y = regression_data
    search = GridSearchCV(
        DummyRegressor(strategy="constant"),
        param_grid={"constant": [0.1, 1.0, 10.0]},
        cv=2,
    )
    report = evaluate(search, X, y)
    report.estimator_.best_params_ = {"constant": "not-a-number"}
    codes = set(report.checks.summarize().frame(section="issue")["code"])
    assert "SKD014" not in codes


@pytest.mark.parametrize(
    "search",
    [
        GridSearchCV(
            DummyRegressor(strategy="constant"),
            param_grid={"constant": [0.1, 1.0, 10.0]},
            cv=2,
        ),
        RandomizedSearchCV(
            DummyRegressor(strategy="constant"),
            param_distributions={"constant": [0.1, 1.0, 10.0]},
            cv=2,
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:The total space of parameters .* is smaller than n_iter:UserWarning"
)
def test_search_classes(regression_data, search):
    """SKD014 runs for GridSearchCV and RandomizedSearchCV using cv_results_."""
    X, y = regression_data
    report = evaluate(search, X, y)
    report.estimator_.best_params_ = {"constant": 0.1}
    issues = report.checks.summarize().frame(section="issue").set_index("code")
    assert "SKD014" in issues.index
    assert "minimum" in issues.loc["SKD014", "explanation"]


@pytest.mark.parametrize(
    "search, best_params",
    [
        (
            GridSearchCV(Ridge(), param_grid={"alpha": [0.0, 1.0, 10.0]}, cv=2),
            {"alpha": 0.0},
        ),
        (
            GridSearchCV(
                Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
                param_grid={"ridge__alpha": [0.0, 1.0, 10.0]},
                cv=2,
            ),
            {"ridge__alpha": 0.0},
        ),
    ],
)
def test_not_raised_when_search_edge_matches_space_edge(
    regression_data, search, best_params
):
    """SKD014 is absent when the search minimum equals the parameter-space minimum."""
    X, y = regression_data
    report = evaluate(search, X, y)
    report.estimator_.best_params_ = best_params
    codes = set(report.checks.summarize().frame(section="issue")["code"])
    assert "SKD014" not in codes


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.filterwarnings("ignore:Linear regression models with a zero l1")
def test_not_raised_when_maximum_matches_space_edge(regression_data):
    """SKD014 is absent when the tried maximum equals the closed space upper bound."""
    X, y = regression_data
    search = GridSearchCV(
        ElasticNet(max_iter=5000),
        param_grid={"l1_ratio": [0.0, 0.5, 1.0]},
        cv=2,
    )
    report = evaluate(search, X, y)
    report.estimator_.best_params_ = {"l1_ratio": 1.0}
    codes = set(report.checks.summarize().frame(section="issue")["code"])
    assert "SKD014" not in codes


@pytest.mark.parametrize("report_type", ["estimator", "cross-validation"])
def test_not_applicable_for_plain_estimator(report_type, regression_data):
    """
    SKD014 raises CheckNotApplicable when the report estimator isn't a BaseSearchCV.
    """
    X, y = regression_data
    report = evaluate(
        DummyRegressor(),
        X,
        y,
        splitter=0.2 if report_type == "estimator" else 3,
    )
    with pytest.raises(CheckNotApplicable, match="not a BaseSearchCV"):
        CheckHyperparamsAtSearchEdge().check_function(report)
