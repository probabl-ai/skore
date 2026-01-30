from importlib import import_module
from importlib.metadata import EntryPoint, EntryPoints
from re import escape
from sys import modules
from unittest.mock import Mock, patch

from pytest import fixture, raises, warns


@fixture
def FakeLogin():
    def login(*, arg1=None, arg2=None):
        pass

    return Mock(spec=login, return_value=None)


class FakeEntryPoint(EntryPoint):
    def load(self):
        return self.value


def test_warning_old_joblib():
    function = Mock()

    with (
        patch.dict("sys.modules"),
        patch("joblib.__version__", "1.3.0"),
        patch("skore._config.set_config", function),
        warns(UserWarning, match="Because your version of joblib is older than 1.4"),
    ):
        modules.pop("skore", None)
        import_module("skore")

    function.assert_called_with(show_progress=False)


def test_login_local(monkeypatch, FakeLogin):
    monkeypatch.setattr(
        "skore.entry_points",
        lambda **kwargs: EntryPoints(
            [
                FakeEntryPoint(
                    name="local",
                    value=FakeLogin,
                    group="skore.plugins.login",
                )
            ]
        ),
    )

    from skore import login

    login(mode="local", arg1=1, arg2=2)

    assert not FakeLogin.called


def test_login_hub(monkeypatch, FakeLogin):
    monkeypatch.setattr(
        "skore.entry_points",
        lambda **kwargs: EntryPoints(
            [
                FakeEntryPoint(
                    name="hub",
                    value=FakeLogin,
                    group="skore.plugins.login",
                )
            ]
        ),
    )

    from skore import login

    login(arg1=1, arg2=2)

    assert FakeLogin.called
    assert not FakeLogin.call_args.args
    assert FakeLogin.call_args.kwargs == {"arg1": 1, "arg2": 2}


def test_login_hub_without_plugin(monkeypatch):
    monkeypatch.setattr("skore.entry_points", lambda **kwargs: EntryPoints([]))

    from skore import login

    with raises(
        ValueError, match=escape("Unknown mode `hub`. Please install `skore[hub]`.")
    ):
        login(arg1=1, arg2=2)
