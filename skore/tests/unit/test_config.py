from concurrent.futures import ThreadPoolExecutor

from joblib import Parallel, parallel_config
from pytest import mark, raises

from skore import configuration
from skore._utils.parallel import delayed
from skore._utils.testing import _change_configuration_for_testing


class CustomException(Exception): ...


def test_configuration_show_progress():
    assert configuration.show_progress is True

    configuration.show_progress = False

    assert configuration.show_progress is False


def test_configuration_plot_backend():
    assert configuration.plot_backend == "matplotlib"

    configuration.plot_backend = "plotly"

    assert configuration.plot_backend == "plotly"


def test_configuration_ignore_checks():
    assert configuration.ignore_checks is None

    configuration.ignore_checks = ["SKD001", "SKD002"]

    assert configuration.ignore_checks == ["SKD001", "SKD002"]
    configuration.ignore_checks = None


def test_ignore_checks_is_copied_from_the_caller():
    """Mutating the list passed in must not change the stored codes."""
    codes = ["SKD001"]
    configuration.ignore_checks = codes
    codes.append("SKD002")

    assert configuration.ignore_checks == ["SKD001"]
    configuration.ignore_checks = None


def test_ignore_checks_context_drops_in_place_changes():
    """An append inside the context manager does not survive the block."""
    configuration.ignore_checks = ["SKD001"]
    try:
        with configuration(show_progress=False):
            configuration.ignore_checks.append("LEAK")
            assert configuration.ignore_checks == ["SKD001", "LEAK"]
        assert configuration.ignore_checks == ["SKD001"]
    finally:
        configuration.ignore_checks = None
        configuration.show_progress = True


def test_ignore_checks_rejects_a_string_and_non_strings():
    with raises(TypeError, match="list or tuple of strings"):
        configuration.ignore_checks = "SK0093"
    with raises(TypeError, match="items must be strings"):
        configuration.ignore_checks = [1]
    assert configuration.ignore_checks is None


def test_ignore_checks_worker_has_its_own_list():
    """A threading task that appends must not change the main thread's list."""
    configuration.ignore_checks = ["SKD001"]

    def append_ignore_check():
        configuration.ignore_checks.append("FROM_WORKER")
        return list(configuration.ignore_checks)

    try:
        wrapper, args, kwargs = delayed(append_ignore_check)()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(wrapper, *args, **kwargs).result()
        assert result == ["SKD001", "FROM_WORKER"]
        assert configuration.ignore_checks == ["SKD001"]
    finally:
        configuration.ignore_checks = None


def test_configuration_call():
    assert configuration.show_progress is True
    assert configuration.plot_backend == "matplotlib"
    assert configuration.ignore_checks is None

    with configuration():
        assert configuration.show_progress is True
        assert configuration.plot_backend == "matplotlib"

    assert configuration.show_progress is True
    assert configuration.plot_backend == "matplotlib"
    assert configuration.ignore_checks is None

    with configuration(show_progress=False):
        assert configuration.show_progress is False
        assert configuration.plot_backend == "matplotlib"

    assert configuration.show_progress is True
    assert configuration.plot_backend == "matplotlib"

    with configuration(plot_backend="plotly"):
        assert configuration.show_progress is True
        assert configuration.plot_backend == "plotly"

    assert configuration.show_progress is True
    assert configuration.plot_backend == "matplotlib"

    with configuration(show_progress=False):
        assert configuration.show_progress is False
        assert configuration.plot_backend == "matplotlib"

        with configuration(plot_backend="plotly"):
            assert configuration.show_progress is False
            assert configuration.plot_backend == "plotly"

    assert configuration.show_progress is True
    assert configuration.plot_backend == "matplotlib"

    with raises(CustomException), configuration(show_progress=False):
        assert configuration.show_progress is False
        assert configuration.plot_backend == "matplotlib"

        raise CustomException()

    assert configuration.show_progress is True
    assert configuration.plot_backend == "matplotlib"
    assert configuration.ignore_checks is None

    with configuration(ignore_checks=["SKD001"]):
        assert configuration.ignore_checks == ["SKD001"]

    assert configuration.ignore_checks is None


@mark.parametrize("backend", ["loky", "multiprocessing", "threading"])
def test_configuration_through_joblib(backend):
    """
    Using joblib, ensure that:
    - processes can't modify main process's configuration,
    - processes inherit main process's configuration,
    - tasks don't impact each other (one process can be shared between tasks in joblib).
    """

    assert configuration.show_progress is True
    assert configuration.plot_backend == "matplotlib"

    # with default configuration
    with parallel_config(backend=backend):
        tasks = Parallel(n_jobs=2)(
            delayed(_change_configuration_for_testing)() for _ in range(4)
        )

        assert tasks == 4 * [
            (
                (True, "matplotlib"),
                ("show_progress_thread", "plot_backend_thread"),
            ),
        ]

    # ensure there is no impact on main process's configuration
    assert configuration.show_progress is True
    assert configuration.plot_backend == "matplotlib"

    # with modified configuration
    configuration.show_progress = False
    configuration.plot_backend = "plotly"

    with parallel_config(backend=backend):
        tasks = Parallel(n_jobs=2)(
            delayed(_change_configuration_for_testing)() for _ in range(4)
        )

        assert tasks == 4 * [
            (
                (False, "plotly"),
                ("show_progress_thread", "plot_backend_thread"),
            ),
        ]

    # ensure there is no impact on main process's configuration
    assert configuration.show_progress is False
    assert configuration.plot_backend == "plotly"


def test_configuration_through_threading():
    """
    Using threading from stdlib, ensure that:
    - thread can't modify main thread's configuration,
    - thread inherits main thread's configuration.
    """
    with ThreadPoolExecutor() as executor:
        assert configuration.show_progress is True
        assert configuration.plot_backend == "matplotlib"

        # with default configuration
        assert executor.submit(_change_configuration_for_testing).result() == (
            (True, "matplotlib"),
            ("show_progress_thread", "plot_backend_thread"),
        )

        # ensure there is no impact on main thread's configuration
        assert configuration.show_progress is True
        assert configuration.plot_backend == "matplotlib"

        # with modified configuration
        configuration.show_progress = False
        configuration.plot_backend = "plotly"

        assert executor.submit(_change_configuration_for_testing).result() == (
            (False, "plotly"),
            ("show_progress_thread", "plot_backend_thread"),
        )

        # ensure there is no impact on main thread's configuration
        assert configuration.show_progress is False
        assert configuration.plot_backend == "plotly"
