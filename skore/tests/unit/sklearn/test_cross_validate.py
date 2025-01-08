import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from skore.sklearn.cross_validation import CrossValidationReporter
from skore.sklearn.cross_validation.cross_validation_helpers import _get_scorers_to_add


def prepare_cv():
    from sklearn import datasets, linear_model

    diabetes = datasets.load_diabetes()
    X = diabetes.data[:150]
    y = diabetes.target[:150]
    lasso = linear_model.Lasso()
    return lasso, X, y


def test_cross_validate():
    """When the user doesn't pass `return_estimator=True` or `return_indices=True`,
    the user-facing results don't have the associated keys but the internal ones do."""
    lasso, X, y = prepare_cv()
    reporter = CrossValidationReporter(lasso, X, y, cv=3)

    assert "estimator" not in reporter.cv_results
    assert "indices" not in reporter.cv_results
    assert "estimator" in reporter._cv_results
    assert "indices" in reporter._cv_results


def test_cross_validate_return_estimator():
    """When the user passes `return_estimator=True` and `return_indices=True`,
    the user-facing results have the associated keys and the internal ones do."""
    lasso, X, y = prepare_cv()
    reporter = CrossValidationReporter(
        lasso, X, y, cv=3, return_estimator=True, return_indices=True
    )

    assert "estimator" in reporter.cv_results
    assert "indices" in reporter.cv_results
    assert "estimator" in reporter._cv_results
    assert "indices" in reporter._cv_results


@pytest.mark.parametrize(
    "estimator,dataset_func,dataset_kwargs,expected_keys",
    [
        # Regression case
        (
            LinearRegression(),
            make_regression,
            {"n_targets": 1},
            {"r2", "root_mean_squared_error"},
        ),
        # Binary classification with predict_proba
        (
            LogisticRegression(),
            make_classification,
            {"n_classes": 2},
            {"recall", "precision", "roc_auc", "brier_score_loss"},
        ),
        # Binary classification without predict_proba
        (
            SVC(probability=False),
            make_classification,
            {"n_classes": 2},
            {"recall", "precision", "roc_auc"},
        ),
        # Multiclass classification with predict_proba
        (
            LogisticRegression(),
            make_classification,
            {"n_classes": 3, "n_clusters_per_class": 1},
            {
                "recall_weighted",
                "precision_weighted",
                "roc_auc_ovr_weighted",
                "log_loss",
            },
        ),
        # Multiclass classification without predict_proba
        (
            SVC(probability=False),
            make_classification,
            {"n_classes": 3, "n_clusters_per_class": 1},
            {"recall_weighted", "precision_weighted"},
        ),
    ],
    ids=[
        "regression",
        "binary_classification_with_proba",
        "binary_classification_without_proba",
        "multiclass_with_proba",
        "multiclass_without_proba",
    ],
)
def test_get_scorers_to_add(estimator, dataset_func, dataset_kwargs, expected_keys):
    # Generate data
    X, y = dataset_func(**dataset_kwargs)

    # Get scorers
    scorers = _get_scorers_to_add(estimator, y)

    # Check that we have exactly the expected keys
    assert set(scorers.keys()) == expected_keys
