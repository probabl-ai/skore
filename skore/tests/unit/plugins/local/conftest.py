from datetime import UTC, datetime

from pytest import fixture


@fixture
def now():
    return datetime.now(tz=UTC)


@fixture
def nowstr(now):
    return now.isoformat()


@fixture
def Datetime(now):
    now_from_fixture = now

    class Datetime:
        nows = []
        nows_isoformat = []

        def __init__(self, *args, **kwargs): ...

        @staticmethod
        def now(*args, **kwargs):
            now = datetime.now(tz=UTC) if Datetime.nows else now_from_fixture
            now_isoformat = now.isoformat()

            Datetime.nows.append(now)
            Datetime.nows_isoformat.append(now_isoformat)

            return now

    return Datetime


def pytest_configure(config):
    import matplotlib

    # Use a non-interactive ``matplotlib.backend`` that can only write to files.
    #
    # https://github.com/matplotlib/matplotlib/issues/29119
    # https://matplotlib.org/stable/users/explain/figure/backends.html#selecting-a-backend
    matplotlib.use("agg")


@fixture
def monkeypatch_skrub(monkeypatch):
    """
    Make `skrub.TableReport.html_snippet()` reproducible

    https://github.com/skrub-data/skrub/blob/35f573ce586fe61ef2c72f4c0c4b188ebf2e664b/skrub/_reporting/_html.py#L153
    """
    monkeypatch.setattr("secrets.token_hex", lambda: "<token>")


@fixture
def monkeypatch_matplotlib(monkeypatch):
    """
    Make `matplotlib.Figure.savefig(format="svg")` reproducible

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


@fixture(autouse=True)
def setup(
    monkeypatch_matplotlib,
    monkeypatch_skrub,
): ...
