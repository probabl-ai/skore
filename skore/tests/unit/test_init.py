from importlib import import_module
from importlib.metadata import version
from sys import modules
from unittest.mock import Mock

from pytest import warns

import skore


def test_warning_old_joblib(monkeypatch):
    configuration = Mock()

    monkeypatch.setattr("joblib.__version__", "1.3.0")
    monkeypatch.setattr("skore._config.configuration", configuration)
    monkeypatch.delitem(modules, "skore")

    with warns(UserWarning, match="Because your version of joblib is older than 1.4"):
        import_module("skore")

    assert hasattr(configuration, "show_progress") is True
    assert configuration.show_progress is False


def test__version__returns_version():
    assert skore.__version__ == version("skore")
