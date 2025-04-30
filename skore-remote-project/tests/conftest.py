from datetime import datetime, timezone

from pytest import fixture


@fixture
def now():
    return datetime.now(tz=timezone.utc)


@fixture
def nowstr(now):
    return now.isoformat()


@fixture(autouse=True)
def setup(monkeypatch, tmp_path):
    import importlib
    import tempfile

    import matplotlib

    with monkeypatch.context() as mp:
        # Change ``TMPDIR`` used by ``tempfile.gettempdir()`` to point to ``tmp_path``,
        # so that it is automatically deleted after use, with no impact on the user's
        # environment.
        mp.setenv("TMPDIR", str(tmp_path))

        # Force the reload of the ``tempfile`` module to change the cached return of
        # ``tempfile.gettempdir()``.
        #
        # https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir
        importlib.reload(tempfile)

        # Make `skrub.TableReport.html_snippet()` reproducible
        #
        # https://github.com/skrub-data/skrub/blob/35f573ce586fe61ef2c72f4c0c4b188ebf2e664b/skrub/_reporting/_html.py#L153
        mp.setattr("secrets.token_hex", lambda: "<token>")

        # Make `matplotlib.Figure.savefig(format="svg")` reproducible
        #
        # https://matplotlib.org/stable/users/prev_whats_new/whats_new_2.1.0.html#reproducible-ps-pdf-and-svg-output
        # https://matplotlib.org/stable/users/prev_whats_new/whats_new_3.10.0.html#svg-id-rcparam
        mp.setenv("SOURCE_DATE_EPOCH", "0")

        matplotlib_rcparams = matplotlib.rcParams.copy()

        try:
            matplotlib.rcParams["svg.hashsalt"] = "<hashsalt>"

            if "svg.id" in matplotlib.rcParams:
                matplotlib.rcParams["svg.id"] = "<id>"

            # Use a non-interactive ``matplotlib.backend`` that can only write to files.
            #
            # https://github.com/matplotlib/matplotlib/issues/29119
            # https://matplotlib.org/stable/users/explain/figure/backends.html#selecting-a-backend
            matplotlib.rcParams["backend"] = "agg"

            yield
        finally:
            matplotlib.rcParams = matplotlib_rcparams
