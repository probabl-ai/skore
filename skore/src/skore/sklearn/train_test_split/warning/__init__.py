"""Train-test split warnings.

This module implements the warnings that can be reported to the user when using
`train_test_split`.
"""

from .high_class_imbalance_warning import (
    HighClassImbalanceWarning,
)

__all__ = [
    HighClassImbalanceWarning.__name__,
]
