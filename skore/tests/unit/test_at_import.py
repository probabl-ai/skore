import importlib
from unittest.mock import patch

import pytest


def test_warning_old_joblib():
    """Test that importing skore with old joblib version raises a warning."""
    with (
        patch("joblib.__version__", "1.3.0"),
        pytest.warns(
            UserWarning, match="Because your version of joblib is older than 1.4"
        ),
    ):
        # Use importlib instead of raw `import`, to force re-importing
        importlib.reload(importlib.import_module("skore"))
