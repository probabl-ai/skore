from skore.cross_validate import cross_validate
from skore.item.cross_validate_item import CrossValidateItem


def test_cross_validate():
    from sklearn import datasets, linear_model

    diabetes = datasets.load_diabetes()
    X = diabetes.data[:150]
    y = diabetes.target[:150]
    lasso = linear_model.Lasso()
    cv_results = cross_validate(lasso, X, y, cv=3)

    assert isinstance(cv_results, CrossValidateItem)
