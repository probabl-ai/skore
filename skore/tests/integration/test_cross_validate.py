import numpy
import pandas
import pytest
import sklearn.model_selection
from numpy import array
from sklearn import datasets, linear_model
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from skore.cross_validate import cross_validate
from skore.item.cross_validation_item import (
    CrossValidationAggregationItem,
    CrossValidationItem,
    plot_cross_validation,
)


@pytest.fixture
def rf():
    iris = datasets.load_iris()
    X = iris.data[:150]
    y = numpy.random.randint(2, size=150)
    rf = RandomForestClassifier()
    return rf, X, y


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
    def test_scorer(self, rf, in_memory_project, scoring):
        cv_results = cross_validate(*rf, scoring=scoring, project=in_memory_project)
        cv_results_sklearn = sklearn.model_selection.cross_validate(
            *rf, scoring=scoring
        )

        assert isinstance(
            in_memory_project.get_item("cross_validation"), CrossValidationItem
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
    def test_data_type(self, rf, in_memory_project, convert_args):
        args = convert_args(*rf)

        cv_results = cross_validate(*args, project=in_memory_project)
        cv_results_sklearn = sklearn.model_selection.cross_validate(*args)

        assert isinstance(
            in_memory_project.get_item("cross_validation"), CrossValidationItem
        )
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
    def test_cross_validate(self, in_memory_project, get_args):
        args = get_args(self)
        cv_results = cross_validate(*args, project=in_memory_project)
        cv_results_sklearn = sklearn.model_selection.cross_validate(*args)

        assert isinstance(
            in_memory_project.get_item("cross_validation"), CrossValidationItem
        )
        assert set(cv_results.keys()).issuperset(cv_results_sklearn.keys())
        assert all(len(v) == 5 for v in cv_results.values())


def test_plot_cross_validation():
    cv_results = {
        "fit_time": array([0.00058246, 0.00041819, 0.00039363]),
        "score_time": array([0.00101399, 0.00072646, 0.00072432]),
        "test_score": array([0.3315057, 0.08022103, 0.03531816]),
        "test_r2": array([0.3315057, 0.08022103, 0.03531816]),
        "test_neg_mean_squared_error": array(
            [-3635.52042005, -3573.35050281, -6114.77901585]
        ),
    }
    plot_cross_validation(cv_results)


def test_aggregated_cross_validation(rf, in_memory_project):
    args = rf
    cross_validate(*args, project=in_memory_project)
    assert isinstance(
        in_memory_project.item_repository.get_item("cross_validation_aggregated"),
        CrossValidationAggregationItem,
    )

    cross_validate(*args, project=in_memory_project)
    cross_validate(*args, project=in_memory_project)
    cross_validate(*args, project=in_memory_project)
    in_memory_project.item_repository.get_item(
        "cross_validation_aggregated"
    ).plot.show()
