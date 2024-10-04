from skore.cross_validate import cross_validate


def test_cross_validate(in_memory_project):
    from sklearn import datasets, linear_model

    diabetes = datasets.load_diabetes()
    X = diabetes.data[:150]
    y = diabetes.target[:150]
    lasso = linear_model.Lasso()

    in_memory_project.put(
        "my_cross_val", cross_validate(lasso, X, y, cv=3), on_error="raise"
    )
    breakpoint()
