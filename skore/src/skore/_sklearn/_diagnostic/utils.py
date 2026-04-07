_TIMING_METRICS = {
    "Fit time (s)",
    "Predict time (s)",
    "fit time (s)",
    "predict time (s)",
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
    favorability: str,
    floor: float,
    fraction: float,
) -> bool:
    """Check whether `score` is significantly better than `baseline`.

    The gap threshold is `fraction` of the reference score, floored at `floor`
    to prevent the threshold from vanishing on near-zero scores.
    """
    if favorability == "(↗︎)":
        return score - baseline >= adaptive_threshold(
            floor=floor, fraction=fraction, references=(baseline,)
        )
    if favorability == "(↘︎)":
        return baseline - score >= adaptive_threshold(
            floor=floor, fraction=fraction, references=(baseline,)
        )
    return False


def majority_vote(votes: list[bool]) -> tuple[bool, int, int]:
    """Apply a strict-majority rule to `votes`.

    Returns ``(majority, n_positive, n_total)``.
    """
    n_positive = sum(votes)
    total = len(votes)
    return n_positive > total / 2, n_positive, total


class DiagnosticNotApplicable(Exception):
    """Raised when a check cannot run on the given report."""


def validate_check_result(result: object) -> None:
    """Validate the return value of a diagnostic check function.

    Parameters
    ----------
    result : object
        the value returned by a check callable

    Raises
    ------
    TypeError
        If the result is not a ``dict[str, dict]`` with required keys.
    """
    if not isinstance(result, dict):
        raise TypeError(
            f"Check function must return a dict, got {type(result).__name__!r}."
        )
    for code, issue in result.items():
        if not isinstance(code, str):
            raise TypeError(
                f"Check code must be a string, got {type(code).__name__!r}."
            )
        if not isinstance(issue, dict):
            raise TypeError(
                f"Issue for code {code!r} must be a dict, got {type(issue).__name__!r}."
            )
        for required in ("title", "explanation"):
            if required not in issue:
                raise TypeError(
                    f"Issue for code {code!r} is missing required key {required!r}."
                )
