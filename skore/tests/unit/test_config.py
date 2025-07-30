from concurrent.futures import ThreadPoolExecutor

import pytest
from skore import config_context, get_config, set_config
from skore._config import _set_show_progress_for_testing
from skore._utils._parallel import Parallel, delayed


def test_config_context():
    assert get_config() == {
        "show_progress": True,
        "plot_backend": "matplotlib",
    }

    # Not using as a context manager affects nothing
    config_context(show_progress=False)
    assert get_config()["show_progress"] is True

    with config_context(show_progress=False):
        assert get_config() == {
            "show_progress": False,
            "plot_backend": "matplotlib",
        }
    assert get_config()["show_progress"] is True

    with config_context(show_progress=False):
        with config_context(show_progress=None):
            assert get_config()["show_progress"] is False

        assert get_config()["show_progress"] is False

        with config_context(show_progress=None):
            assert get_config()["show_progress"] is False

            with config_context(show_progress=None):
                assert get_config()["show_progress"] is False

                # global setting will not be retained outside of context that
                # did not modify this setting
                set_config(show_progress=True)
                assert get_config()["show_progress"] is True

            assert get_config()["show_progress"] is False

        assert get_config()["show_progress"] is False

    assert get_config() == {
        "show_progress": True,
        "plot_backend": "matplotlib",
    }

    # No positional arguments
    with pytest.raises(TypeError):
        config_context(True)

    # No unknown arguments
    with pytest.raises(TypeError):
        config_context(do_something_else=True).__enter__()


def test_config_context_exception():
    assert get_config()["show_progress"] is True
    try:
        with config_context(show_progress=False):
            assert get_config()["show_progress"] is False
            raise ValueError()
    except ValueError:
        pass
    assert get_config()["show_progress"] is True


def test_set_config():
    assert get_config()["show_progress"] is True
    set_config(show_progress=None)
    assert get_config()["show_progress"] is True
    set_config(show_progress=False)
    assert get_config()["show_progress"] is False
    set_config(show_progress=None)
    assert get_config()["show_progress"] is False
    set_config(show_progress=None)
    assert get_config()["show_progress"] is False

    # No unknown arguments
    with pytest.raises(TypeError):
        set_config(do_something_else=True)

    # reset the context to default for other tests
    set_config(show_progress=True)
    assert get_config()["show_progress"] is True


@pytest.mark.parametrize("backend", ["loky", "multiprocessing", "threading"])
def test_config_threadsafe_joblib(backend):
    """Test that the global config is threadsafe with all joblib backends.
    Two jobs are spawned and each sets `show_progress` to two different values.
    When the job with a duration of 0.1s completes, the `show_progress` value
    should be the same as the value passed to the function. In other words,
    it is not influenced by the other job setting `show_progress` to True.
    """
    show_progresses = [False, True, False, True]
    sleep_durations = [0.1, 0.2, 0.1, 0.2]

    items = Parallel(backend=backend, n_jobs=2)(
        delayed(_set_show_progress_for_testing)(show_progress, sleep_duration)
        for show_progress, sleep_duration in zip(
            show_progresses, sleep_durations, strict=False
        )
    )

    assert items == [False, True, False, True]


def test_config_threadsafe():
    """Uses threads directly to test that the global config does not change
    between threads. Same test as `test_config_threadsafe_joblib` but with
    `ThreadPoolExecutor`."""

    show_progresses = [False, True, False, True]
    sleep_durations = [0.1, 0.2, 0.1, 0.2]

    with ThreadPoolExecutor(max_workers=2) as e:
        items = [
            output
            for output in e.map(
                _set_show_progress_for_testing, show_progresses, sleep_durations
            )
        ]

    assert items == [False, True, False, True]
