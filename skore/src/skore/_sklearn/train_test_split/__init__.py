"""Train-test split.

This module implements `train_test_split`, a wrapper over scikit-learn's own
function of the same name. This wrapper adds some functionality in the form
of warnings shown to the user when common issues are detected, e.g. imbalanced
data, time-series data...

It also exposes :class:`TrainTestSplit`, a single train-test splitter that
implements the cross-validation protocol.
"""

from skore._sklearn.train_test_split.train_test_split import TrainTestSplit

__all__ = ["TrainTestSplit"]
