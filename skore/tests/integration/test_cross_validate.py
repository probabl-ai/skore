import numpy
import pytest
import sklearn.model_selection
from numpy import array
from sklearn import datasets, linear_model
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from skore.cross_validate import cross_validate
from skore.item.cross_validate_item import CrossValidationItem, plot_cross_validation


@pytest.fixture()
def lasso():
    diabetes = datasets.load_diabetes()
    X = diabetes.data[:150]
    y = diabetes.target[:150]
    lasso = linear_model.Lasso()
    return lasso, X, y


def test_cross_validate_regression(in_memory_project, lasso):
    args = lasso
    kwargs = {"cv": 3}

    cv_results = cross_validate(*args, **kwargs, project=in_memory_project)
    cv_results_sklearn = sklearn.model_selection.cross_validate(*args, **kwargs)

    assert isinstance(
        in_memory_project.get_item("cross_validation"), CrossValidationItem
    )
    assert cv_results.keys() == cv_results_sklearn.keys()
    assert all(len(v) == kwargs["cv"] for v in cv_results.values())


def test_cross_validate_regression_extra_metrics(in_memory_project, lasso):
    args = list(lasso)
    kwargs = {"scoring": "r2", "cv": 3}

    cv_results = cross_validate(*args, **kwargs, project=in_memory_project)
    cv_results_sklearn = sklearn.model_selection.cross_validate(*args, **kwargs)

    assert isinstance(
        in_memory_project.get_item("cross_validation"), CrossValidationItem
    )
    assert set(cv_results.keys()).issuperset(cv_results_sklearn.keys())
    assert all(len(v) == kwargs["cv"] for v in cv_results.values())


def test_cross_validate_2_extra_metrics(in_memory_project, lasso):
    args = list(lasso)
    kwargs = {"scoring": ["r2", "neg_mean_squared_error"], "cv": 3}

    cv_results = cross_validate(*args, **kwargs, project=in_memory_project)
    cv_results_sklearn = sklearn.model_selection.cross_validate(*args, **kwargs)

    assert isinstance(
        in_memory_project.get_item("cross_validation"), CrossValidationItem
    )
    assert cv_results.keys() == cv_results_sklearn.keys()
    assert all(len(v) == kwargs["cv"] for v in cv_results.values())


def test_cross_validation_multi_class_classification_sub_estimator(in_memory_project):
    X, y = datasets.load_iris(return_X_y=True)
    model = OneVsOneClassifier(LinearSVC())

    args = [model, X, y]
    kwargs = {"cv": 3}

    cv_results = cross_validate(*args, **kwargs, project=in_memory_project)
    cv_results_sklearn = sklearn.model_selection.cross_validate(*args, **kwargs)

    assert isinstance(
        in_memory_project.get_item("cross_validation"), CrossValidationItem
    )
    assert cv_results.keys() == cv_results_sklearn.keys()
    assert all(len(v) == kwargs["cv"] for v in cv_results.values())


def test_cross_validation_multi_class_classification(in_memory_project):
    iris = datasets.load_iris()
    X = iris.data[:250]
    y = iris.target[:250]
    rf = RandomForestClassifier()

    args = [rf, X, y]
    kwargs = {"cv": 3}

    cv_results = cross_validate(*args, **kwargs, project=in_memory_project)
    cv_results_sklearn = sklearn.model_selection.cross_validate(*args, **kwargs)

    assert isinstance(
        in_memory_project.get_item("cross_validation"), CrossValidationItem
    )
    assert cv_results.keys() == cv_results_sklearn.keys()
    assert all(len(v) == kwargs["cv"] for v in cv_results.values())


def test_cross_validation_binary_classification(in_memory_project):
    iris = datasets.load_iris()
    X = iris.data[:150]
    y = numpy.random.randint(2, size=150)
    rf = RandomForestClassifier()

    args = [rf, X, y]
    kwargs = {"cv": 3}

    cv_results = cross_validate(*args, **kwargs, project=in_memory_project)
    cv_results_sklearn = sklearn.model_selection.cross_validate(*args, **kwargs)

    assert isinstance(
        in_memory_project.get_item("cross_validation"), CrossValidationItem
    )
    assert cv_results.keys() == cv_results_sklearn.keys()
    assert all(len(v) == kwargs["cv"] for v in cv_results.values())


def test_cross_validation_clustering(in_memory_project):
    iris = datasets.load_iris()
    X = iris.data[:150]
    kmeans = KMeans()

    args = [kmeans, X]
    kwargs = {"cv": 3}

    cv_results = cross_validate(*args, **kwargs, project=in_memory_project)
    cv_results_sklearn = sklearn.model_selection.cross_validate(*args, **kwargs)

    assert isinstance(
        in_memory_project.get_item("cross_validation"), CrossValidationItem
    )
    assert cv_results.keys() == cv_results_sklearn.keys()
    assert all(len(v) == kwargs["cv"] for v in cv_results.values())


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
