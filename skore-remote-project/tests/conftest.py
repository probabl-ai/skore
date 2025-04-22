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
