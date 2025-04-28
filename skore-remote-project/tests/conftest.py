from datetime import datetime, timezone

from pytest import fixture


@fixture
def now():
    return datetime.now(tz=timezone.utc)


@fixture
def nowstr(now):
    return now.isoformat()


@fixture(autouse=True)
def monkeypatch_tmpdir(monkeypatch, tmp_path):
    """
    A pytest fixture that temporarily changes the system's temporary directory.

    It modifies the ``TMPDIR`` environment variable to point to a temporary path managed
    by pytest's ``tmp_path`` fixture. Thus, it will be automatically deleted after use
    and will not impact the user's environment.

    It also forces the reload of the ``tempfile`` module to change the cached return of
    ``tempfile.gettempdir()``.
    """
    import tempfile
    from importlib import reload

    with monkeypatch.context() as mp:
        mp.setenv("TMPDIR", str(tmp_path))

        # The first call of `tempfile.gettempdir` being cached, reload the module.
        # https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir
        reload(tempfile)

        yield


@fixture
def reproducible(monkeypatch):
    import matplotlib

    # Make `skrub.TableReport.html_snippet()` reproducible
    # https://github.com/skrub-data/skrub/blob/35f573ce586fe61ef2c72f4c0c4b188ebf2e664b/skrub/_reporting/_html.py#L153
    monkeypatch.setattr("secrets.token_hex", lambda: "<token>")

    # Make `matplotlib.Figure.savefig(format="svg")` reproducible
    # https://matplotlib.org/stable/users/prev_whats_new/whats_new_2.1.0.html#reproducible-ps-pdf-and-svg-output
    # https://matplotlib.org/stable/users/prev_whats_new/whats_new_3.10.0.html#svg-id-rcparam
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")

    try:
        matplotlib.rcParams["svg.hashsalt"] = "<hashsalt>"

        if "svg.id" in matplotlib.rcParams:
            matplotlib.rcParams["svg.id"] = "<id>"

        yield
    finally:
        matplotlib.rcdefaults()
