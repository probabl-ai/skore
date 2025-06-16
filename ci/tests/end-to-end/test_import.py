from pytest import fixture
from sklearn.datasets import make_classification, make_regression, fetch_openml
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from skrub import tabular_learner
from sklearn.model_selection import GridSearchCV


@fixture(scope="module")
def regression():
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return {
        "estimator": LinearRegression(),
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


@fixture(scope="module")
def classification():
    X, y = make_classification(n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return {
        "estimator": LogisticRegression(),
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


@fixture(scope="module")
def gridsearch():
    X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
    y = 1 * (y == ">50K")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return {
        "estimator": GridSearchCV(
            estimator=tabular_learner("classification"),
            param_grid={
                "histgradientboostingclassifier__learning_rate": [0.01, 0.1, 0.2],
                "histgradientboostingclassifier__max_depth": [1, 3, 5],
                "histgradientboostingclassifier__max_leaf_nodes": [30, 60, 90],
            },
            cv=5,
            n_jobs=-1,
            refit=True,
            scoring="neg_log_loss",
        ),
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def test_put_with_local_project(tmp_path, regression, classification, gridsearch):
    import skore
    import skore_local_project

    project = skore.Project("<name>", workspace=tmp_path)

    assert isinstance(project._Project__project, skore_local_project.Project)
    assert project.mode == "local"
    assert project.name == "<name>"
    assert project._Project__project.workspace == tmp_path
    assert project._Project__project.name == "<name>"

    project.put("regression", skore.EstimatorReport(**regression))
    project.put("classification", skore.EstimatorReport(**classification))
    project.put("gridsearch", skore.EstimatorReport(**gridsearch))


def test_simili_put_with_hub_project(regression, classification, gridsearch):
    import skore
    import skore_hub_project

    project = skore.Project("hub://<tenant>/<name>")

    assert isinstance(project._Project__project, skore_hub_project.Project)
    assert project.mode == "hub"
    assert project.name == "<name>"
    assert project._Project__project.tenant == "<tenant>"
    assert project._Project__project.name == "<name>"

    for xp in (regression, classification, gridsearch):
        item = skore_hub_project.item.object_to_item(skore.EstimatorReport(**xp))

        assert item.__metadata__
        assert item.__representation__
        assert item.__parameters__
