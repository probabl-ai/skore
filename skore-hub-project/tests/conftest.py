from datetime import datetime, timezone

from pytest import fixture
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from skore import CrossValidationReport, EstimatorReport


@fixture(scope="module")
def regression() -> EstimatorReport:
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return EstimatorReport(
        Ridge(random_state=42),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@fixture(scope="module")
def cv_regression() -> CrossValidationReport:
    X, y = make_regression(random_state=42)

    return CrossValidationReport(Ridge(random_state=42), X, y)


@fixture(scope="module")
def binary_classification() -> EstimatorReport:
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return EstimatorReport(
        RandomForestClassifier(random_state=42),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


@fixture(scope="module")
def cv_binary_classification() -> CrossValidationReport:
    X, y = make_classification(random_state=42)

    return CrossValidationReport(RandomForestClassifier(random_state=42), X, y)


@fixture
def now():
    return datetime.now(tz=timezone.utc)


@fixture
def nowstr(now):
    return now.isoformat()


def pytest_configure(config):
    import matplotlib

    # Use a non-interactive ``matplotlib.backend`` that can only write to files.
    #
    # https://github.com/matplotlib/matplotlib/issues/29119
    # https://matplotlib.org/stable/users/explain/figure/backends.html#selecting-a-backend
    matplotlib.use("agg")


@fixture
def monkeypatch_tmpdir(monkeypatch, tmp_path):
    """
    Change ``TMPDIR`` used by ``tempfile.gettempdir()`` to point to ``tmp_path``, so
    that it is automatically deleted after use, with no impact on user's environment.

    Force the reload of the ``tempfile`` module to change the cached return of
    ``tempfile.gettempdir()``.

    https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir
    """
    import importlib
    import tempfile

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    importlib.reload(tempfile)


@fixture
def monkeypatch_skrub(monkeypatch):
    """
    Make `skrub.TableReport.html_snippet()` reproducible.

    https://github.com/skrub-data/skrub/blob/35f573ce586fe61ef2c72f4c0c4b188ebf2e664b/skrub/_reporting/_html.py#L153
    """
    monkeypatch.setattr("secrets.token_hex", lambda: "<token>")


@fixture
def monkeypatch_matplotlib(monkeypatch):
    """
    Make `matplotlib.Figure.savefig(format="svg")` reproducible.

    https://matplotlib.org/stable/users/prev_whats_new/whats_new_2.1.0.html#reproducible-ps-pdf-and-svg-output
    https://matplotlib.org/stable/users/prev_whats_new/whats_new_3.10.0.html#svg-id-rcparam
    """
    import matplotlib

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")

    matplotlib_rcparams = matplotlib.rcParams.copy()
    matplotlib.rcParams["svg.hashsalt"] = "<hashsalt>"

    if "svg.id" in matplotlib.rcParams:
        matplotlib.rcParams["svg.id"] = "<id>"

    try:
        yield
    finally:
        matplotlib.rcParams = matplotlib_rcparams


@fixture
def monkeypatch_skore_hub_api_key(monkeypatch):
    """
    Delete `SKORE_HUB_API_KEY` from the environment, to avoid biasing the tests.
    """
    monkeypatch.delenv("SKORE_HUB_API_KEY", raising=False)


@fixture
def monkeypatch_sklearn_estimator_html_repr(monkeypatch):
    """
    Make `sklearn.utils.estimator_html_repr` reproducible.

    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_repr_html/estimator.py#L21
    """
    from importlib.metadata import version

    package_version = [int(number) for number in version("scikit-learn").split(".")]

    if package_version >= [1, 7, 0]:
        importpath = "sklearn.utils._repr_html.estimator._IDCounter.get_id"
    else:
        importpath = "sklearn.utils._estimator_html_repr._IDCounter.get_id"

    monkeypatch.setattr(importpath, lambda self: "<id>")


@fixture(autouse=True)
def setup(
    monkeypatch_tmpdir,
    monkeypatch_matplotlib,
    monkeypatch_skrub,
    monkeypatch_skore_hub_api_key,
    monkeypatch_sklearn_estimator_html_repr,
): ...
