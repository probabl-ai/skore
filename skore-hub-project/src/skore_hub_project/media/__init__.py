from __future__ import annotations

from abc import ABC, abstractmethod
from base64 import b64decode, b64encode
from contextlib import contextmanager
from inspect import signature as inspect_signature
from typing import Any

# from .data import TableReportTest, TableReportTrain
from .feature_importance import (
    Coefficients,
    MeanDecreaseImpurity,
    PermutationTest,
    PermutationTrain,
)
from .model import EstimatorHtmlRepr
from .performance import (
    PrecisionRecallTest,
    PrecisionRecallTrain,
    PredictionErrorTest,
    PredictionErrorTrain,
    RocTest,
    RocTrain,
)

__all__ = [
    "Coefficients",
    "EstimatorHtmlRepr",
    "MeanDecreaseImpurity",
    "PermutationTest",
    "PermutationTrain",
    "PrecisionRecallTest",
    "PrecisionRecallTrain",
    "PredictionErrorTest",
    "PredictionErrorTrain",
    "RocTest",
    "RocTrain",
    # "TableReportTest",
    # "TableReportTrain",
    "b64_str_to_bytes",
    "bytes_to_b64_str",
    "switch_mpl_backend",
]


@contextmanager
def switch_mpl_backend():
    """
    Context-manager for switching ``matplotlib.backend`` to ``agg``.

    Notes
    -----
    The ``agg`` backend is a non-interactive backend that can only write to files.
    It is used in ``skore-hub-project`` to generate artifacts where we don't need an
    X display.

    https://matplotlib.org/stable/users/explain/figure/backends.html#selecting-a-backend
    """
    import matplotlib

    original_backend = matplotlib.get_backend()

    try:
        matplotlib.use("agg")
        yield
    finally:
        matplotlib.use(original_backend)


def bytes_to_b64_str(literal: bytes) -> str:
    """Encode the bytes-like object ``literal`` in a Base64 str."""
    return b64encode(literal).decode("utf-8")


def b64_str_to_bytes(literal: str) -> bytes:
    """Decode the Base64 str object ``literal`` in a bytes."""
    return b64decode(literal.encode("utf-8"))
