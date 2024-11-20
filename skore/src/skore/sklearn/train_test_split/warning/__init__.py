"""Train-test split warnings.

This module implements the warnings that can be reported to the user when using
`train_test_split`.
"""

from .high_class_imbalance_too_few_examples_warning import (
    HighClassImbalanceTooFewExamplesWarning,
)
from .high_class_imbalance_warning import (
    HighClassImbalanceWarning,
)

TRAIN_TEST_SPLIT_WARNINGS = [
    HighClassImbalanceTooFewExamplesWarning,
    HighClassImbalanceWarning,
]

__all__ = [
    "TRAIN_TEST_SPLIT_WARNINGS",
    HighClassImbalanceTooFewExamplesWarning.__name__,
    HighClassImbalanceWarning.__name__,
]
