from unittest.mock import patch

from skore._utils._fixes import _validate_joblib_parallel_params


def test_validate_joblib_parallel_params_old_version():
    """Test that return_as is removed from kwargs when joblib < 1.4."""
    with patch("skore._utils._fixes.joblib.__version__", "1.3.0"):
        kwargs = {"n_jobs": -1, "return_as": "list"}
        result = _validate_joblib_parallel_params(**kwargs)

        assert "return_as" not in result
        assert result == {"n_jobs": -1}


def test_validate_joblib_parallel_params_new_version():
    """Test that return_as is kept in kwargs when joblib >= 1.4."""
    with patch("skore._utils._fixes.joblib.__version__", "1.4.0"):
        kwargs = {"n_jobs": -1, "return_as": "list"}
        result = _validate_joblib_parallel_params(**kwargs)

        assert "return_as" in result
        assert result == kwargs
