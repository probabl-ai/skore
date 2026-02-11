from concurrent.futures import ThreadPoolExecutor
from time import sleep

from pytest import raises

from skore import config_context, get_config
from skore._utils._progress_bar import ProgressBar, track


def test_standalone_progress():
    """Check the general behavior of the progress bar when used standalone."""

    with ProgressBar(description="standalone", total=2) as progress:
        assert progress._progress.live._started is True

        progress.advance()
        progress.advance()

    assert progress.description == "standalone"
    assert progress.total == 2

    assert progress._progress.tasks[0].description == "standalone"
    assert progress._progress.tasks[0].total == 2
    assert progress._progress.live._started is False
    assert progress._progress.finished
    assert progress._progress.tasks[0].finished


def test_nested_progress():
    """Check that we can nest progress bars."""

    with ProgressBar(description="parent", total=1) as parent_progress:
        assert parent_progress._progress.live._started is True

        with ProgressBar(description="child", total=2) as child_progress:
            assert child_progress._progress.live._started is True

            child_progress.advance()
            child_progress.advance()

        parent_progress.advance()

    assert parent_progress != child_progress
    assert parent_progress.description == "parent"
    assert parent_progress.total == 1
    assert child_progress.description == "child"
    assert child_progress.total == 2

    assert parent_progress._progress.tasks[0].description == "parent"
    assert parent_progress._progress.tasks[0].total == 1
    assert parent_progress._progress.live._started is False
    assert parent_progress._progress.finished
    assert parent_progress._progress.tasks[0].finished
    assert child_progress._progress.tasks[0].description == "child"
    assert child_progress._progress.tasks[0].total == 2
    assert child_progress._progress.live._started is False
    assert child_progress._progress.finished
    assert child_progress._progress.tasks[0].finished


def test_exception_handling():
    """Check that the progress bar stops during an exception but in a clean way."""

    class CustomException(Exception): ...

    with raises(CustomException), ProgressBar(description="error", total=2) as progress:
        assert progress._progress.live._started is True

        raise CustomException()

    assert progress.description == "error"
    assert progress.total == 2

    assert progress._progress.tasks[0].description == "error"
    assert progress._progress.tasks[0].total == 2
    assert progress._progress.live._started is False
    assert not progress._progress.finished
    assert not progress._progress.tasks[0].finished


def test_thread_safety():
    """Check thread-safety."""

    t1_progress = None
    t2_progress = None

    def run_t1():
        with ProgressBar(description="t1", total=1) as progress:
            assert progress._progress.live._started is True

            sleep(0.5)

            progress.advance()

        nonlocal t1_progress
        t1_progress = progress
        return "done"

    def run_t2():
        with ProgressBar(description="t2", total=2) as progress:
            assert progress._progress.live._started is True

            progress.advance()
            progress.advance()

        nonlocal t2_progress
        t2_progress = progress
        return "done"

    with ThreadPoolExecutor(max_workers=1) as executor:
        future1 = executor.submit(run_t1)
        future2 = executor.submit(run_t2)

        result1 = future1.result()
        result2 = future2.result()

    assert result1 == "done"
    assert result2 == "done"

    assert t1_progress != t2_progress
    assert t1_progress.description == "t1"
    assert t1_progress.total == 1
    assert t2_progress.description == "t2"
    assert t2_progress.total == 2

    assert t1_progress._progress.tasks[0].description == "t1"
    assert t1_progress._progress.tasks[0].total == 1
    assert t1_progress._progress.live._started is False
    assert t1_progress._progress.finished
    assert t1_progress._progress.tasks[0].finished
    assert t2_progress._progress.tasks[0].description == "t2"
    assert t2_progress._progress.tasks[0].total == 2
    assert t2_progress._progress.live._started is False
    assert t2_progress._progress.finished
    assert t2_progress._progress.tasks[0].finished


def test_track(monkeypatch):
    progress = None

    def RegisteredProgressBar(*args, **kwargs):
        nonlocal progress
        progress = ProgressBar(*args, **kwargs)
        return progress

    monkeypatch.setattr("skore._utils._progress_bar.ProgressBar", RegisteredProgressBar)

    results = list(track(range(10), description="track", total=10))

    assert results == list(range(10))
    assert progress.description == "track"
    assert progress.total == 10

    assert progress._progress.tasks[0].description == "track"
    assert progress._progress.tasks[0].total == 10
    assert progress._progress.live._started is False
    assert progress._progress.finished
    assert progress._progress.tasks[0].finished


def test_disable_progress_bar():
    with ProgressBar(description="progress1", total=0) as progress1:
        assert get_config()["show_progress"] is True
        assert progress1._progress.disable is False

        with (
            config_context(show_progress=False),
            ProgressBar(description="progress2", total=0) as progress2,
        ):
            assert get_config()["show_progress"] is False
            assert progress1._progress.disable is False
            assert progress2._progress.disable is True
