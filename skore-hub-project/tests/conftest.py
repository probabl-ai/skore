from datetime import datetime, timezone
from functools import partial
from importlib import reload
from unittest.mock import Mock
from urllib.parse import urljoin

from httpx import Client, Response
from numpy import array
from pytest import fixture
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from skore import CrossValidationReport, EstimatorReport

from skore_hub_project.project.project import Project


@fixture
def project():
    return Project(workspace="workspace", name="name")


@fixture
def monkeypatch_project_routes(respx_mock):
    mocks = [
        ("get", "/projects/workspace", Response(200)),
        (
            "post",
            "/projects/workspace/name",
            Response(
                200,
                json={
                    "id": 42,
                    "url": "http://domain/workspace/name",
                },
            ),
        ),
    ]

    for method, url, response in mocks:
        respx_mock.request(method=method, url=url).mock(response)


@fixture
def upload_mock():
    from skore_hub_project.artifact.upload import upload

    return Mock(spec=upload, wraps=upload)


@fixture
def monkeypatch_upload_with_mock(monkeypatch, upload_mock):
    monkeypatch.setattr("skore_hub_project.artifact.artifact.upload", upload_mock)


@fixture
def monkeypatch_upload_routes(respx_mock):
    mocks = [
        (
            "post",
            "projects/workspace/name/artifacts",
            Response(201, json=[{"upload_url": "http://chunk1.com/", "chunk_id": 1}]),
        ),
        ("put", "http://chunk1.com", Response(200, headers={"etag": '"<etag1>"'})),
        ("post", "projects/workspace/name/artifacts/complete", Response(200)),
    ]

    for method, url, response in mocks:
        respx_mock.request(method=method, url=url).mock(response)


class FakeClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def request(self, method, url, **kwargs):
        response = super().request(method, urljoin("http://localhost", url), **kwargs)
        response.raise_for_status()

        return response


@fixture
def monkeypatch_project_hub_client(monkeypatch):
    monkeypatch.setattr("skore_hub_project.project.project.HUBClient", FakeClient)


@fixture
def monkeypatch_artifact_hub_client(monkeypatch):
    monkeypatch.setattr("skore_hub_project.artifact.upload.HUBClient", FakeClient)


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
def multiclass_classification() -> EstimatorReport:
    X, y = make_classification(n_classes=3, n_informative=4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return EstimatorReport(
        LogisticRegression(random_state=42),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


@fixture(scope="module")
def cv_binary_classification() -> CrossValidationReport:
    X, y = make_classification(random_state=42)

    return CrossValidationReport(RandomForestClassifier(random_state=42), X, y)


@fixture(scope="module")
def small_cv_binary_classification() -> CrossValidationReport:
    X, y = make_classification(random_state=42, n_samples=10)

    return CrossValidationReport(
        RandomForestClassifier(random_state=42), X, y, splitter=2
    )


def _make_binary_estimator_report_string_labels(*, pos_label=None):
    """Binary classification with string y labels ('negative', 'positive')."""
    X, y = make_classification(random_state=42)
    labels = array(["negative", "positive"])
    y = labels[y]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    report = EstimatorReport(
        RandomForestClassifier(random_state=42),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        pos_label=pos_label,
    )
    return report


def _make_cv_binary_report_string_labels(*, pos_label=None):
    """Cross-validation binary classification with string y labels."""
    X, y = make_classification(random_state=42, n_samples=50)
    labels = array(["negative", "positive"])
    y = labels[y]
    report = CrossValidationReport(
        RandomForestClassifier(random_state=42), X, y, splitter=2, pos_label=pos_label
    )
    return report


@fixture(scope="module")
def binary_classification_string_labels() -> EstimatorReport:
    """Binary classification with string labels, pos_label not set."""
    return _make_binary_estimator_report_string_labels()


@fixture(scope="module")
def binary_classification_string_labels_with_pos_label() -> EstimatorReport:
    """Binary classification with string labels and pos_label='positive'."""
    return _make_binary_estimator_report_string_labels(pos_label="positive")


@fixture(scope="module")
def cv_binary_classification_string_labels() -> CrossValidationReport:
    """CV binary classification with string labels, pos_label not set."""
    return _make_cv_binary_report_string_labels()


@fixture(scope="module")
def cv_binary_classification_string_labels_with_pos_label() -> CrossValidationReport:
    """CV binary classification with string labels and pos_label='positive'."""
    return _make_cv_binary_report_string_labels(pos_label="positive")


@fixture
def now():
    return datetime.now(tz=timezone.utc)


@fixture
def nowstr(now):
    return now.isoformat()


def pytest_configure(config):
    import logging

    logging.getLogger(name="httpx").setLevel(level=logging.WARNING)

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
    import tempfile

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    reload(tempfile)


@fixture
def monkeypatch_skrub(monkeypatch):
    """
    Make `skrub.TableReport.html_snippet()` reproducible.

    https://github.com/skrub-data/skrub/blob/35f573ce586fe61ef2c72f4c0c4b188ebf2e664b/skrub/_reporting/_html.py#L153
    """
    from skrub._dataframe import sample

    monkeypatch.setattr("secrets.token_hex", lambda: "<token>")
    monkeypatch.setattr("skrub._dataframe.sample", partial(sample, seed=42))


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
def monkeypatch_skore_hub_envars(monkeypatch):
    """Delete environment variables that can bias the tests."""
    monkeypatch.delenv("SKORE_HUB_API_KEY", raising=False)
    monkeypatch.delenv("SKORE_HUB_URI", raising=False)


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


@fixture
def monkeypatch_rich(monkeypatch):
    """Make `rich` silent."""
    from rich.console import Console

    monkeypatch.setattr("rich.console.Console", partial(Console, quiet=True))
    monkeypatch.setattr("skore.console.quiet", True)
    monkeypatch.setattr("skore_hub_project.console.quiet", True)


@fixture
def monkeypatch_global_variables(monkeypatch):
    """Reset global variables that can bias the tests."""
    monkeypatch.setattr("skore_hub_project.authentication.login.credentials", None)


@fixture(autouse=True)
def setup(
    monkeypatch_tmpdir,
    monkeypatch_matplotlib,
    monkeypatch_skrub,
    monkeypatch_skore_hub_envars,
    monkeypatch_sklearn_estimator_html_repr,
    monkeypatch_rich,
    monkeypatch_global_variables,
): ...
