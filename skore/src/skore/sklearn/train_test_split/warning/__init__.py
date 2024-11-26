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
from .random_state_unset_warning import RandomStateUnsetWarning
from .shuffle_true_warning import ShuffleTrueWarning
from .stratify_is_set_warning import StratifyWarning
from .time_based_column_warning import TimeBasedColumnWarning

TRAIN_TEST_SPLIT_WARNINGS = [
    HighClassImbalanceTooFewExamplesWarning,
    HighClassImbalanceWarning,
    StratifyWarning,
    RandomStateUnsetWarning,
    ShuffleTrueWarning,
    TimeBasedColumnWarning,
]

__all__ = [
    "TRAIN_TEST_SPLIT_WARNINGS",
    "HighClassImbalanceTooFewExamplesWarning",
    "HighClassImbalanceWarning",
    "StratifyWarning",
    "RandomStateUnsetWarning",
    "ShuffleTrueWarning",
    "TimeBasedColumnWarning",
]
