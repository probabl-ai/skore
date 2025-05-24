import pytest
from skore.utils._progress_bar import progress_decorator


def test_standalone_progress():
    """Check the general behavior of the progress bar when used standalone."""

    class StandaloneTask:
        def __init__(self):
            self._progress_info = None

        @progress_decorator("Standalone Task")
        def run(self, iterations=5):
            progress = self._progress_info["current_progress"]
            task = self._progress_info["current_task"]
            progress.update(task, total=iterations)

            for i in range(iterations):
                progress.update(task, advance=1)
                self._standalone_n_calls = i
            return "done"

    task = StandaloneTask()
    result = task.run()

    assert result == "done"
    assert task._standalone_n_calls == 4
    assert task._progress_info is None


def test_nested_progress():
    """Check that we can nest progress bars."""

    class ParentTask:
        def __init__(self):
            self._progress_info = None

        @progress_decorator("Parent Task")
        def run(self, iterations=3):
            progress = self._progress_info["current_progress"]
            task = self._progress_info["current_task"]
            progress.update(task, total=iterations)

            self._child = ChildTask()
            for i in range(iterations):
                # Share the parent's progress bar with child
                self._child._progress_info = {"current_progress": progress}
                self._child.run()
                progress.update(task, advance=1)
                self._parent_n_calls = i
            return "done"

    class ChildTask:
        def __init__(self):
            self._progress_info = None

        @progress_decorator("Child Task")
        def run(self, iterations=2):
            progress = self._progress_info["current_progress"]
            task = self._progress_info["current_task"]
            progress.update(task, total=iterations)

            for i in range(iterations):
                progress.update(task, advance=1)
                self._child_n_calls = i
            return "done"

    parent = ParentTask()
    result = parent.run()

    assert result == "done"
    assert parent._progress_info is None
    assert parent._parent_n_calls == 2
    assert parent._child._child_n_calls == 1
    assert parent._child._progress_info is None


def test_dynamic_description():
    """Check that we can pass a dynamic description using `self` when calling the
    decorator."""

    class DynamicTask:
        def __init__(self, name):
            self._progress_info = None
            self.name = name

        @progress_decorator(lambda self: f"Processing {self.name}")
        def run(self, iterations=4):
            progress = self._progress_info["current_progress"]
            task = self._progress_info["current_task"]
            progress.update(task, total=iterations)

            for i in range(iterations):
                progress.update(task, advance=1)
                self._dynamic_n_calls = i
            return self.name

    task = DynamicTask("test_task")
    result = task.run()

    assert result == "test_task"
    assert task._progress_info is None
    assert task._dynamic_n_calls == 3


def test_exception_handling():
    """Check that the progress bar stops during an exception but in a clean way."""

    class ErrorTask:
        def __init__(self):
            self._progress_info = None

        @progress_decorator("Error Task")
        def run(self):
            progress = self._progress_info["current_progress"]
            task = self._progress_info["current_task"]
            progress.update(task, total=3)

            progress.update(task, advance=1)
            raise ValueError("Test error")

    task = ErrorTask()
    with pytest.raises(ValueError, match="Test error"):
        task.run()

    # Verify progress bar was cleaned up
    assert task._progress_info is None


def test_child_report_cleanup():
    """Ensure that child reports in reports_ get progress assigned and then cleaned
    up."""

    class Child:
        def __init__(self):
            self._progress_info = None
            self.called = False

        @progress_decorator("Child Process")
        def process(self):
            self.called = True
            return "child_done"

    class Parent:
        def __init__(self):
            self._progress_info = None
            self.reports_ = [Child(), Child()]

        @progress_decorator("Parent Process")
        def run(self):
            results = []
            for rpt in self.reports_:
                results.append(rpt.process())
            return results

    parent = Parent()
    results = parent.run()

    assert results == ["child_done", "child_done"]
    assert all(rp.called for rp in parent.reports_)
    # Verify that progress attributes are cleaned for each child report
    for rp in parent.reports_:
        assert rp._progress_info is None
    # Also verify if parent reports are cleaned up as well
    assert parent._progress_info is None
