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
    """A pytest fixture that temporarily changes the system's temporary directory.

    This fixture modifies the TMPDIR environment variable to point to a temporary
    path provided by pytest's tmp_path fixture. It also handles the cached nature
    of tempfile.gettempdir() by reloading the tempfile module.
    """
    import tempfile
    from importlib import reload

    with monkeypatch.context() as mp:
        mp.setenv("TMPDIR", str(tmp_path))

        # The first call of `tempfile.gettempdir` being cached, reload the module.
        # https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir
        reload(tempfile)

        yield
