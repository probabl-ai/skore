import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.dummy import DummyClassifier

from skore import CrossValidationReport, EstimatorReport
from skore._utils._dataframe import (
    _normalize_X_as_dataframe,
    _normalize_y_as_dataframe,
)


def test_normalize_X_as_dataframe_sparse():
    """Check that _normalize_X_as_dataframe rejects sparse matrices."""
    X_sparse = sp.csr_matrix(np.random.rand(10, 2))
    with pytest.raises(NotImplementedError, match="not supported for sparse matrices"):
        _normalize_X_as_dataframe(X_sparse)


def test_normalize_y_as_dataframe_sparse():
    """Check that _normalize_y_as_dataframe rejects sparse matrices."""
    y_sparse = sp.csr_matrix(np.random.rand(10, 1))
    with pytest.raises(NotImplementedError, match="not supported for sparse matrices"):
        _normalize_y_as_dataframe(y_sparse)


def test_estimator_report_data_analyze_sparse_X():
    """Check EstimatorReport graceful degradation for sparse X."""
    X_sparse = sp.csr_matrix(np.random.rand(10, 2))
    y_dense = np.random.randint(0, 2, size=10)

    # Initialization succeeds
    report = EstimatorReport(
        DummyClassifier(strategy="prior"), X_train=X_sparse, y_train=y_dense
    )

    # Analysis triggers our specific error
    with pytest.raises(NotImplementedError, match="not supported for sparse matrices"):
        report.data.analyze(data_source="train")


def test_estimator_report_data_analyze_sparse_y():
    """Check EstimatorReport graceful degradation for sparse y."""
    X_dense = np.random.rand(10, 2)
    # Even if we pass sparse y, we set fit=False so it doesn't crash during init
    y_sparse = sp.csr_matrix(np.random.randint(0, 2, size=(10, 1)))

    # We must skip fitting because sklearn estimators generally reject sparse y
    report = EstimatorReport(
        DummyClassifier(strategy="prior"), fit=False, X_train=X_dense, y_train=y_sparse
    )

    with pytest.raises(NotImplementedError, match="not supported for sparse matrices"):
        report.data.analyze(data_source="train")


def test_cv_report_data_analyze_sparse_X():
    """Check CrossValidationReport graceful degradation for sparse X."""
    X_sparse = sp.csr_matrix(np.random.rand(10, 2))
    y_dense = np.random.randint(0, 2, size=10)

    report = CrossValidationReport(
        DummyClassifier(strategy="prior"), X=X_sparse, y=y_dense, splitter=2
    )

    with pytest.raises(NotImplementedError, match="not supported for sparse matrices"):
        report.data.analyze()


def test_cv_report_data_analyze_sparse_y():
    """Check CrossValidationReport graceful degradation for sparse y."""
    X_dense = np.random.rand(10, 2)
    y_sparse = sp.csr_matrix(np.random.randint(0, 2, size=(10, 1)))

    # Because CVReport fits internally and sklearn generally rejects sparse y,
    # we test that the initialization throws the expected sklearn TypeError.
    msg = "Sparse data was passed"
    with pytest.raises(TypeError, match=msg):
        CrossValidationReport(
            DummyClassifier(strategy="prior"), X=X_dense, y=y_sparse, splitter=2
        )
