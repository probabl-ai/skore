from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

from pytest import raises

from skore._utils._progress_bar import progress_decorator


def test_standalone_progress():
    """Check the general behavior of the progress bar when used standalone."""

    class StandaloneTask:
        @progress_decorator("Standalone")
        def run(self, *, progress):
            assert progress.total is None
            assert progress._progress.live._started is True
            assert progress._progress.tasks[0].total is None

            progress.total = 2
            progress.advance()
            progress.advance()

            self.progress = progress
            return "done"

    task = StandaloneTask()
    result = task.run()

    assert result == "done"

    assert task.progress.description == "Standalone"
    assert task.progress.total == 2

    assert task.progress._progress.tasks[0].description == "Standalone"
    assert task.progress._progress.tasks[0].total == 2
    assert task.progress._progress.live._started is False
    assert task.progress._progress.finished
    assert task.progress._progress.tasks[0].finished


def test_nested_progress():
    """Check that we can nest progress bars."""

    parent_progress = None
    child_progress = None

    class ParentTask:
        @progress_decorator("Parent")
        def run(self, *, progress):
            assert progress.total is None
            assert progress._progress.live._started is True
            assert progress._progress.tasks[0].total is None

            progress.total = 1
            ChildTask().run()
            progress.advance()

            nonlocal parent_progress
            parent_progress = progress
            return "done"

    class ChildTask:
        @progress_decorator("Child")
        def run(self, *, progress):
            assert progress.total is None
            assert progress._progress.live._started is True
            assert progress._progress.tasks[0].total is None

            progress.total = 2
            progress.advance()
            progress.advance()

            nonlocal child_progress
            child_progress = progress
            return "done"

    parent = ParentTask()
    result = parent.run()

    assert result == "done"

    assert parent_progress != child_progress
    assert parent_progress.description == "Parent"
    assert parent_progress.total == 1
    assert child_progress.description == "Child"
    assert child_progress.total == 2

    assert parent_progress._progress.tasks[0].description == "Parent"
    assert parent_progress._progress.tasks[0].total == 1
    assert parent_progress._progress.live._started is False
    assert parent_progress._progress.finished
    assert parent_progress._progress.tasks[0].finished
    assert child_progress._progress.tasks[0].description == "Child"
    assert child_progress._progress.tasks[0].total == 2
    assert child_progress._progress.live._started is False
    assert child_progress._progress.finished
    assert child_progress._progress.tasks[0].finished


def test_dynamic_description():
    """Check that we can pass a dynamic description using `self` when calling the
    decorator."""

    class DynamicTask:
        def __init__(self, name):
            self.name = name

        @progress_decorator(lambda self: f"{self.name}")
        def run(self, *, progress):
            assert progress.total is None
            assert progress._progress.live._started is True
            assert progress._progress.tasks[0].total is None

            progress.total = 2
            progress.advance()
            progress.advance()

            self.progress = progress
            return "done"

    task = DynamicTask("test")
    result = task.run()

    assert result == "done"

    assert task.progress.description == "test"
    assert task.progress.total == 2

    assert task.progress._progress.tasks[0].description == "test"
    assert task.progress._progress.tasks[0].total == 2
    assert task.progress._progress.live._started is False
    assert task.progress._progress.finished
    assert task.progress._progress.tasks[0].finished


def test_exception_handling():
    """Check that the progress bar stops during an exception but in a clean way."""

    class ErrorTask:
        @progress_decorator("Error")
        def run(self, *, progress):
            assert progress.total is None
            assert progress._progress.live._started is True
            assert progress._progress.tasks[0].total is None

            self.progress = progress
            raise ValueError("Test error")

    task = ErrorTask()

    with raises(ValueError, match="Test error"):
        task.run()

    assert task.progress.description == "Error"
    assert task.progress.total is None

    assert task.progress._progress.tasks[0].description == "Error"
    assert task.progress._progress.tasks[0].total is None
    assert task.progress._progress.live._started is False
    assert not task.progress._progress.finished
    assert not task.progress._progress.tasks[0].finished


def test_thread_safety():
    """Check thread-safety."""

    t1_progress = None
    t2_progress = None

    class T1Task:
        @progress_decorator("T1")
        def run(self, *, progress):
            assert progress.total is None
            assert progress._progress.live._started is True
            assert progress._progress.tasks[0].total is None

            sleep(0.5)

            progress.total = 1
            progress.advance()

            nonlocal t1_progress
            t1_progress = progress
            return "done"

    class T2Task:
        @progress_decorator("T2")
        def run(self, *, progress):
            assert progress.total is None
            assert progress._progress.live._started is True
            assert progress._progress.tasks[0].total is None

            progress.total = 2
            progress.advance()
            progress.advance()

            nonlocal t2_progress
            t2_progress = progress
            return "done"

    with ThreadPoolExecutor(max_workers=1) as executor:
        future1 = executor.submit(lambda: T1Task().run())
        future2 = executor.submit(lambda: T2Task().run())

        result1 = future1.result()
        result2 = future2.result()

    assert result1 == "done"
    assert result2 == "done"

    assert t1_progress != t2_progress
    assert t1_progress.description == "T1"
    assert t1_progress.total == 1
    assert t2_progress.description == "T2"
    assert t2_progress.total == 2

    assert t1_progress._progress.tasks[0].description == "T1"
    assert t1_progress._progress.tasks[0].total == 1
    assert t1_progress._progress.live._started is False
    assert t1_progress._progress.finished
    assert t1_progress._progress.tasks[0].finished
    assert t2_progress._progress.tasks[0].description == "T2"
    assert t2_progress._progress.tasks[0].total == 2
    assert t2_progress._progress.live._started is False
    assert t2_progress._progress.finished
    assert t2_progress._progress.tasks[0].finished
