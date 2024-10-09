import sklearn.model_selection
from skore.cross_validate import cross_validate, plot_cross_validation
from skore.item.cross_validate_item import CrossValidateItem


def test_cross_validate(in_memory_project):
    from sklearn import datasets, linear_model

    diabetes = datasets.load_diabetes()
    X = diabetes.data[:150]
    y = diabetes.target[:150]
    lasso = linear_model.Lasso()

    n_splits = 3
    cv_results = cross_validate(lasso, X, y, cv=n_splits, project=in_memory_project)
    cv_results_sklearn = sklearn.model_selection.cross_validate(
        lasso, X, y, cv=n_splits
    )

    assert isinstance(in_memory_project.get_item("cross_validation"), CrossValidateItem)
    assert cv_results.keys() == cv_results_sklearn.keys()
    assert all(len(v) == n_splits for v in cv_results.values())


def test_plot_cross_validation():
    from numpy import array

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
