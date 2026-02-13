from concurrent.futures import ThreadPoolExecutor

from joblib import Parallel, parallel_config
from pytest import mark, raises

from skore import configuration
from skore._config import _change_configuration_for_testing
from skore._utils._parallel import delayed


class CustomException(Exception): ...


def test_configuration_show_progress():
    assert configuration.show_progress is True

    configuration.show_progress = False

    assert configuration.show_progress is False


def test_configuration_plot_backend():
    assert configuration.plot_backend == "matplotlib"

    configuration.plot_backend = "plotly"

    assert configuration.plot_backend == "plotly"


def test_configuration_call():
    assert configuration.show_progress is True
    assert configuration.plot_backend == "matplotlib"

    with configuration():
        assert configuration.show_progress is True
        assert configuration.plot_backend == "matplotlib"

    assert configuration.show_progress is True
    assert configuration.plot_backend == "matplotlib"

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
