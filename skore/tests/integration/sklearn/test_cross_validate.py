import numpy
import pandas
import pytest
import sklearn.model_selection
from sklearn import datasets, linear_model
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from skore import CrossValidationReporter
from skore.item.cross_validation_item import CrossValidationItem


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


def prepare_cv():
    from sklearn import datasets, linear_model

    diabetes = datasets.load_diabetes()
    X = diabetes.data[:150]
    y = diabetes.target[:150]
    lasso = linear_model.Lasso()

    return lasso, X, y


def test_put_cross_validation_reporter(in_memory_project):
    project = in_memory_project

    lasso, X, y = prepare_cv()
    reporter = CrossValidationReporter(lasso, X, y, cv=3)

    project.put("cross-validation", reporter)

    assert isinstance(project.get_item("cross-validation"), CrossValidationItem)
