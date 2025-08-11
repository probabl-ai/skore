from unittest.mock import patch

import pytest
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


def test_skore_check_is_fitted():
    """Test that not fitted models raise NotFittedError"""
    from sklearn.base import BaseEstimator
    from sklearn.exceptions import NotFittedError
    from sklearn.linear_model import LogisticRegression
    from skore._utils._fixes import skore_check_is_fitted

    class FakeSkorchModel(BaseEstimator):
        def __init__(self):
            self.history_ = []
            self.initialized_ = True
            self.virtual_params_ = {}

        def fit(self):
            self.init_context_ = {}
            self.callbacks_ = []
            self.prefixes_ = []
            self.cuda_dependent_attributes_ = []
            self.module_ = object()
            self.criterion_ = object()
            self.optimizer_ = object()
            self.classes_inferred_ = []

    sklearn_not_fitted = LogisticRegression()
    with pytest.raises(NotFittedError):
        skore_check_is_fitted(sklearn_not_fitted)

    sklearn_fitted = LogisticRegression().fit([[0, 0], [1, 1]], [0, 1])
    skore_check_is_fitted(sklearn_fitted)

    skorch_not_fitted = FakeSkorchModel()
    with pytest.raises(NotFittedError):
        skore_check_is_fitted(skorch_not_fitted)

    skorch_fitted = FakeSkorchModel()
    skorch_fitted.fit()
    skore_check_is_fitted(skorch_fitted)
