import numpy
import pandas
import pytest
import sklearn.model_selection
from sklearn import datasets, linear_model
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from skore import CrossValidationReporter
from skore.persistence.item import CrossValidationReporterItem
from skore.sklearn.cross_validation.cross_validation_helpers import _get_scorers_to_add


@pytest.fixture
def rf():
    iris = datasets.load_iris()
    X = iris.data[:100]
    y = numpy.random.randint(2, size=100)
    rf = RandomForestClassifier()
    return rf, X, y


@pytest.fixture
def fake_cross_validate(monkeypatch):
    def _fake_cross_validate(*args, **kwargs):
        result = {"test_score": [1] * 5, "score_time": [1] * 5, "fit_time": [1] * 5}
        if kwargs.get("return_estimator"):
            result["estimator"] = []
        if kwargs.get("return_indices"):
            result["indices"] = {"train": [[1] * 5] * 5, "test": [[1] * 5] * 5}
        if kwargs.get("return_train_score"):
            result["train_score"] = [1] * 5
        return result

    monkeypatch.setattr("sklearn.model_selection.cross_validate", _fake_cross_validate)


class TestInputScorers:
    """Test that cross_validate works regardless of the scorers data type."""

    def confusion_matrix(clf, X, y):
        from sklearn.metrics import confusion_matrix

        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        return {"tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]}

    def true_positive(clf, X, y):
        from sklearn.metrics import confusion_matrix

        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        true_positive = cm[1, 1]
        return true_positive

    @pytest.mark.parametrize(
        "scoring",
        [
            confusion_matrix,  # callable returning dict
            true_positive,  # callable returning float
            {"true_positive": true_positive, "accuracy": "accuracy"},  # dict
            ["accuracy", "f1"],  # list of strings
            ("accuracy", "f1"),  # tuple of strings
            "accuracy",  # string
            None,
        ],
    )
    def test_scorer(self, rf, in_memory_project, scoring, fake_cross_validate):
        reporter = CrossValidationReporter(*rf, scoring=scoring)
        cv_results = reporter.cv_results
        cv_results_sklearn = sklearn.model_selection.cross_validate(
            *rf, scoring=scoring
        )

        assert set(cv_results.keys()).issuperset(cv_results_sklearn.keys())
        assert all(len(v) == 5 for v in cv_results.values())


class TestInputDataType:
    """Test that cross_validate works regardless of the input data type."""

    def data_is_list(model, X, y):
        return model, X.tolist(), y.tolist()

    def data_is_pandas(model, X, y):
        return model, pandas.DataFrame(X), pandas.Series(y)

    @pytest.mark.parametrize("convert_args", [data_is_list, data_is_pandas])
    def test_data_type(self, rf, in_memory_project, convert_args, fake_cross_validate):
        args = convert_args(*rf)

        reporter = CrossValidationReporter(*args)
        cv_results = reporter.cv_results
        cv_results_sklearn = sklearn.model_selection.cross_validate(*args)

        assert set(cv_results.keys()).issuperset(cv_results_sklearn.keys())
        assert all(len(v) == 5 for v in cv_results.values())


class TestMLTask:
    """Test that cross_validate works regardless of the ML task/estimator."""

    def iris_X_y(self):
        iris = datasets.load_iris()
        X = iris.data[:150]
        y = iris.target[:150]
        return X, y

    def binary_classification(self):
        X, _ = self.iris_X_y()
        y = numpy.random.randint(2, size=150)
        return RandomForestClassifier(), X, y

    def multiclass_classification(self):
        X, y = self.iris_X_y()
        return RandomForestClassifier(), X, y

    def multiclass_classification_no_predict_proba(self):
        X, y = self.iris_X_y()
        return SVC(), X, y

    def multiclass_classification_sub_estimator(self):
        X, y = self.iris_X_y()
        return OneVsOneClassifier(SVC()), X, y

    def regression(self):
        X, y = datasets.load_diabetes(return_X_y=True)
        X = X[:150]
        y = y[:150]
        return linear_model.Lasso(), X, y

    def clustering(self):
        X, _ = self.iris_X_y()
        return KMeans(), X

    @pytest.mark.parametrize(
        "get_args",
        [
            binary_classification,
            multiclass_classification,
            multiclass_classification_no_predict_proba,
            multiclass_classification_sub_estimator,
            regression,
            clustering,
        ],
    )
    def test_cross_validate(self, in_memory_project, get_args, fake_cross_validate):
        args = get_args(self)

        reporter = CrossValidationReporter(*args)
        cv_results = reporter.cv_results
        cv_results_sklearn = sklearn.model_selection.cross_validate(*args)

        assert set(cv_results.keys()).issuperset(cv_results_sklearn.keys())
        assert all(len(v) == 5 for v in cv_results.values())


@pytest.fixture
def binary_classifier():
    X, y = datasets.make_classification(n_classes=2, random_state=42)
    return LogisticRegression(), X, y


@pytest.fixture
def multiclass_classifier():
    X, y = datasets.make_classification(
        n_classes=3, n_clusters_per_class=1, random_state=42
    )
    return LogisticRegression(), X, y


@pytest.fixture
def single_output_regression():
    X, y = datasets.make_regression(n_targets=1, random_state=42)
    return LinearRegression(), X, y


@pytest.fixture
def multi_output_regression():
    X, y = datasets.make_regression(n_targets=2, random_state=42)
    return LinearRegression(), X, y


@pytest.mark.parametrize(
    "fixture_name",
    [
        "binary_classifier",
        "multiclass_classifier",
        "single_output_regression",
        "multi_output_regression",
    ],
)
def test_cross_validation_reporter(in_memory_project, fixture_name, request):
    """Check that we can serialize the `CrossValidationReporter` and retrieve it."""
    model, X, y = request.getfixturevalue(fixture_name)
    reporter = CrossValidationReporter(model, X, y, cv=3)

    in_memory_project.put("cross-validation", reporter)

    retrieved_item = in_memory_project.item_repository.get_item("cross-validation")
    assert isinstance(retrieved_item, CrossValidationReporterItem)


@pytest.mark.parametrize(
    "fixture_name",
    [
        "binary_classifier",
        "multiclass_classifier",
        "single_output_regression",
        "multi_output_regression",
    ],
)
def test_cross_validation_reporter_equivalence_cross_validate(
    in_memory_project, fixture_name, request
):
    """Check that we have an equivalent result to `cross_validate`."""
    # mapping between the scorers names in skore and in sklearn
    map_skore_to_sklearn = {
        "r2": "r2",
        "root_mean_squared_error": "neg_root_mean_squared_error",
        "roc_auc": "roc_auc",
        "brier_score_loss": "neg_brier_score",
        "recall": "recall",
        "precision": "precision",
        "recall_weighted": "recall_weighted",
        "precision_weighted": "precision_weighted",
        "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
        "log_loss": "neg_log_loss",
    }
    model, X, y = request.getfixturevalue(fixture_name)
    reporter = CrossValidationReporter(
        model, X, y, cv=3, return_estimator=True, return_train_score=True
    )

    scorers_used_skore = _get_scorers_to_add(model, y)
    scorers_sklearn = [map_skore_to_sklearn[k] for k in scorers_used_skore]
    cv_results_sklearn = cross_validate(
        model,
        X,
        y,
        cv=3,
        scoring=scorers_sklearn,
        return_estimator=True,
        return_train_score=True,
    )

    # check the equivalence between the scores
    for scorer_skore_name in scorers_used_skore:
        for type_set in ["test", "train"]:
            score_skore = reporter._cv_results[f"{type_set}_{scorer_skore_name}"]
            score_sklearn = cv_results_sklearn[
                f"{type_set}_{map_skore_to_sklearn[scorer_skore_name]}"
            ]
            if map_skore_to_sklearn[scorer_skore_name].startswith("neg_"):
                numpy.testing.assert_allclose(score_skore, -score_sklearn)
            else:
                numpy.testing.assert_allclose(score_skore, score_sklearn)
