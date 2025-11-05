"""Tests for search_cv patching functions."""

from sklearn.model_selection import GridSearchCV, _search

from skore._sklearn.search_cv import patched_fit_and_score, patched_format_results
from skore._utils._patch import patch_function, patch_instance_method


def test_patched_fit_and_score_and_format_results_add_extra_keys(
    logistic_binary_classification_data,
):
    """Test that patched functions add estimator, train_indices, and test_indices."""
    estimator, X, y = logistic_binary_classification_data

    param_grid = {"C": [0.1, 1.0, 10.0]}
    search = GridSearchCV(estimator, param_grid)

    with (
        patch_function(_search, "_fit_and_score", patched_fit_and_score),
        patch_instance_method(search, "_format_results", patched_format_results),
    ):
        search.fit(X, y)

    expected_new_keys = ("estimator", "train_indices", "test_indices")
    for key in expected_new_keys:
        assert key in search.cv_results_

    search.fit(X, y)
    for key in expected_new_keys:
        assert key not in search.cv_results_
