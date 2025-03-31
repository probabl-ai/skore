import sys
from unittest.mock import patch

import pytest


def test_warning_old_joblib():
    """Test that importing skore with old joblib version raises a warning."""
    # Remove skore from sys.modules to force a fresh import
    if "skore" in sys.modules:
        del sys.modules["skore"]

    with (
        patch("joblib.__version__", "1.3.0"),
        pytest.warns(
            UserWarning, match="Because your version of joblib is older than 1.4"
        ),
    ):
        import skore  # noqa: F401
