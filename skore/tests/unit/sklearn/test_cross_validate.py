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
        pytest.param(
            LinearRegression(),
            make_regression,
            {"n_targets": 1},
            {"r2", "root_mean_squared_error"},
            id="regression",
        ),
        pytest.param(
            LogisticRegression(),
            make_classification,
            {"n_classes": 2},
            {"recall", "precision", "roc_auc", "brier_score_loss"},
            id="binary_classification_with_proba",
        ),
        pytest.param(
            SVC(probability=False),
            make_classification,
            {"n_classes": 2},
            {"recall", "precision", "roc_auc"},
            id="binary_classification_without_proba",
        ),
        pytest.param(
            LogisticRegression(),
            make_classification,
            {"n_classes": 3, "n_clusters_per_class": 1},
            {
                "recall_weighted",
                "precision_weighted",
                "roc_auc_ovr_weighted",
                "log_loss",
            },
            id="multiclass_with_proba",
        ),
        pytest.param(
            SVC(probability=False),
            make_classification,
            {"n_classes": 3, "n_clusters_per_class": 1},
            {"recall_weighted", "precision_weighted"},
            id="multiclass_without_proba",
        ),
    ],
)
def test_get_scorers_to_add(estimator, dataset_func, dataset_kwargs, expected_keys):
    """Check that the scorers to add are correct.

    Non-regression test for:
    https://github.com/probabl-ai/skore/issues/1050
    """
    X, y = dataset_func(**dataset_kwargs)
    scorers = _get_scorers_to_add(estimator, y)
    assert set(scorers.keys()) == expected_keys
