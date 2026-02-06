"""Unit tests for ``skore._utils._testing``."""

import numpy as np
import pytest

from skore._utils._testing import (
    MockAccessor,
    MockDisplay,
    MockEstimator,
    MockReport,
)


def test_mock_estimator_fit_predict():
    """MockEstimator.fit sets classes_, predict returns ones."""
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 0])
    est = MockEstimator(error=ValueError("test"))
    est.fit(X, y)
    assert list(est.classes_) == [0, 1]
    pred = est.predict(X)
    assert pred.shape == (3,)
    assert np.all(pred == 1)


def test_mock_estimator_sklearn_clone_increments_n_call():
    """MockEstimator.__sklearn_clone__ increments n_call and returns self."""
    est = MockEstimator(error=RuntimeError("test"), n_call=0)
    assert est.n_call == 0
    out = est.__sklearn_clone__()
    assert out is est
    assert est.n_call == 1


def test_mock_estimator_fit_raises_when_n_call_exceeds_fail_after_n_clone():
    """MockEstimator.fit raises error when n_call > fail_after_n_clone."""
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    error = RuntimeError("clone limit")
    est = MockEstimator(error=error, n_call=0, fail_after_n_clone=2)
    est.__sklearn_clone__()
    est.__sklearn_clone__()
    est.__sklearn_clone__()
    assert est.n_call == 3
    with pytest.raises(RuntimeError, match="clone limit"):
        est.fit(X, y)


@pytest.fixture
def mock_estimator():
    """Minimal estimator for MockReport."""
    est = MockEstimator(error=ValueError("unused"))
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    est.fit(X, y)
    return est


def test_mock_report_init_and_properties(mock_estimator):
    """MockReport stores estimator and data, exposes properties."""
    X_train = np.array([[0], [1]])
    y_train = np.array([0, 1])
    X_test = np.array([[2]])
    y_test = np.array([1])
    report = MockReport(
        mock_estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert report.estimator_ is mock_estimator
    assert report.X_train is X_train
    assert report.y_train is y_train
    assert report.X_test is X_test
    assert report.y_test is y_test
    assert report.no_private == "no_private"
    assert report.attr_without_description == "attr_without_description"


def test_mock_report_get_help_title():
    """MockReport._get_help_title returns 'Mock report'."""
    est = MockEstimator(error=ValueError("unused"))
    report = MockReport(est)
    assert report._get_help_title() == "Mock report"


def test_mock_report_accessor_config():
    """MockReport._ACCESSOR_CONFIG is empty."""
    assert MockReport._ACCESSOR_CONFIG == {}


def test_mock_accessor_init_and_verbose_name(mock_estimator):
    """MockAccessor stores parent, has _verbose_name."""
    report = MockReport(mock_estimator)
    accessor = MockAccessor(parent=report)
    assert accessor._parent is report
    assert MockAccessor._verbose_name == "mock_accessor"


def test_mock_accessor_get_help_tree_title():
    """MockAccessor._get_help_tree_title returns 'Mock accessor'."""
    est = MockEstimator(error=ValueError("unused"))
    report = MockReport(est)
    accessor = MockAccessor(parent=report)
    assert accessor._get_help_tree_title() == "Mock accessor"


def test_mock_display_plot():
    """MockDisplay.plot accepts kwargs and returns None."""
    disp = MockDisplay()
    assert disp.plot() is None
    assert disp.plot(a=1, b=2) is None


def test_mock_display_set_style():
    """MockDisplay.set_style accepts policy and kwargs."""
    disp = MockDisplay()
    assert disp.set_style() is None
    assert disp.set_style(policy="override") is None


def test_mock_display_frame():
    """MockDisplay.frame returns empty DataFrame."""
    disp = MockDisplay()
    out = disp.frame()
    assert out.empty
    assert list(out.columns) == []


def test_mock_display_help():
    """MockDisplay.help returns None."""
    disp = MockDisplay()
    assert disp.help() is None
