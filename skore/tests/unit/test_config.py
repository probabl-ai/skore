import pytest
from skore import config_context, get_config, set_config


def test_config_context():
    assert get_config() == {
        "show_progress": True,
    }

    # Not using as a context manager affects nothing
    config_context(show_progress=False)
    assert get_config()["show_progress"] is True

    with config_context(show_progress=False):
        assert get_config() == {
            "show_progress": False,
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
