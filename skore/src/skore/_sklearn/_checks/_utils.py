from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    from skore._sklearn._estimator.report import EstimatorReport

_TIMING_METRICS = {
    "Fit time (s)",
    "Predict time (s)",
    "fit_time_s",
    "predict_time_s",
    "Fit time s",
    "Predict time s",
}


def adaptive_threshold(
    *, floor: float, fraction: float, references: tuple[float, ...]
) -> float:
    """Compute a scale-aware threshold.

    Returns ``max(floor, fraction * abs(references))``. The floor
    prevents the threshold from vanishing on near-zero scores; scaling by
    the reference magnitude keeps it meaningful for large-valued metrics.
    """
    return max(floor, fraction * max(abs(reference) for reference in references))


def check_score_gap_to_baseline(
    score: float,
    baseline: float,
    greater_is_better: bool | None,
    floor: float,
    fraction: float,
) -> bool:
    """Check whether `score` is significantly better than `baseline`.

    The gap threshold is `fraction` of the reference score, floored at `floor`
    to prevent the threshold from vanishing on near-zero scores.
    """
    if pd.isna(greater_is_better):
        return False

    if greater_is_better:
        return score - baseline >= adaptive_threshold(
            floor=floor, fraction=fraction, references=(baseline,)
        )
    return baseline - score >= adaptive_threshold(
        floor=floor, fraction=fraction, references=(baseline,)
    )


def majority_vote(votes: list[bool]) -> tuple[bool, int, int]:
    """Apply a strict-majority rule to `votes`.

    Returns ``(majority, n_positive, n_total)``.
    """
    n_positive = sum(votes)
    total = len(votes)
    return n_positive > total / 2, n_positive, total


def detect_outliers_modified_zscore(scores, threshold=3):
    """Detect outliers using the modified Z-score method.

    The constant 0.6745 is a scaling factor that makes the MAD a consistent estimator
    of the standard deviation for Gaussian data, so that the resulting
    scores are comparable to ordinary Z-scores.

    See https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    if mad == 0:
        return np.zeros_like(scores)
    modified_z_scores = 0.6745 * (scores - median) / mad

    return np.abs(modified_z_scores) > threshold


class CheckNotApplicable(Exception):
    """Raised when a check cannot run on the given report."""


def split_preprocessor_estimator(estimator):
    """Return ``(preprocessor, predictor)`` from a possibly wrapped estimator.

    Splits sklearn :class:`~sklearn.pipeline.Pipeline` into its preprocessing
    steps and final predictor.
    """
    if isinstance(estimator, Pipeline):
        return estimator[:-1], estimator[-1]
    return None, estimator


def get_preprocessed_data(
    report: EstimatorReport,
    *,
    target: Literal["X", "y"] = "X",
    concatenate: bool = False,
) -> np.ndarray | pd.DataFrame | None:
    """Return feature matrix or target vector from a report.

    When ``target == "X"`` and the report's estimator is a
    :class:`~sklearn.pipeline.Pipeline`, the raw feature matrix is passed
    through the fitted preprocessor (all steps except the last) before being
    returned, so the result reflects what the predictor actually sees.

    Returns ``None`` when no data is available or when the preprocessor
    produces an unsupported type (e.g. sparse matrices).

    Parameters
    ----------
    report : _BaseReport
        the report to extract data from.

    target : {"X", "y"}
        whether to return the feature matrix or the target vector.

    concatenate : bool
        when true and both train and test are available, return their
        concatenation. Otherwise return train if available, else test.

    Returns
    -------
    np.ndarray, pd.DataFrame, or None
    """
    train = report.X_train if target == "X" else report.y_train
    test = report.X_test if target == "X" else report.y_test

    if concatenate and train is not None and test is not None:
        data = (
            pd.concat([train, test], axis=0, ignore_index=True)
            if isinstance(train, pd.DataFrame)
            else np.concatenate([train, test])
        )
    elif train is not None:
        data = train
    elif test is not None:
        data = test
    else:
        return None

    if target == "X":
        preprocessor, _ = split_preprocessor_estimator(report.estimator_)
        if preprocessor is not None and len(preprocessor.steps) > 0:
            data = preprocessor.transform(data)

    if not isinstance(data, (np.ndarray, pd.DataFrame)):
        return None
    return data
