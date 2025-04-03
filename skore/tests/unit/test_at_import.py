import importlib
import sys
from unittest.mock import Mock, patch

from pytest import warns


def test_warning_old_joblib():
    function = Mock()

    with (
        patch.dict("sys.modules"),
        patch("joblib.__version__", "1.3.0"),
        patch("skore._config.set_config", function),
        warns(UserWarning, match="Because your version of joblib is older than 1.4"),
    ):
        sys.modules.pop("skore", None)
        importlib.import_module("skore")

    function.assert_called_with(show_progress=False)
