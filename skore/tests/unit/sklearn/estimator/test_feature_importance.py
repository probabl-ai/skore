import pandas as pd
import pytest
import sklearn.linear_model
from sklearn.base import is_classifier, is_regressor
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from skore import EstimatorReport


@pytest.fixture
def regression_data():
    return make_regression(n_features=5, random_state=42)


@pytest.fixture
def classification_data():
    return make_classification(n_features=5, random_state=42)


def test_estimator_report_feature_importance_help(capsys, regression_data):
    """Check that the help method writes to the console."""
    X, y = regression_data
    report = EstimatorReport(LinearRegression().fit(X, y), X_test=X, y_test=y)

    report.feature_importance.help()
    captured = capsys.readouterr()
    assert "Available feature importance methods" in captured.out
    assert "coefficients" in captured.out


def test_estimator_report_feature_importance_repr(regression_data):
    """Check that __repr__ returns a string starting with the expected prefix."""
    X, y = regression_data
    report = EstimatorReport(LinearRegression().fit(X, y), X_test=X, y_test=y)

    repr_str = repr(report.feature_importance)
    assert "skore.EstimatorReport.feature_importance" in repr_str
    assert "report.feature_importance.help()" in repr_str


########################################################################################
# Check the coefficients feature importance metric
########################################################################################


@pytest.mark.parametrize(
    "data, estimator, column_base_name, expected_shape",
    [
        (
            make_regression(n_features=5, random_state=42),
            LinearRegression(),
            None,
            (6, 1),
        ),
        (
            make_classification(n_features=5, random_state=42),
            LogisticRegression(),
            None,
            (6, 1),
        ),
        (
            make_classification(
                n_features=5,
                n_classes=3,
                n_samples=30,
                n_informative=3,
                random_state=42,
            ),
            LogisticRegression(),
            "Class",
            (6, 3),
        ),
        (
            make_classification(
                n_features=5,
                n_classes=3,
                n_samples=30,
                n_informative=3,
                random_state=42,
            ),
            make_pipeline(StandardScaler(), LogisticRegression()),
            "Class",
            (6, 3),
        ),
        (
            make_regression(n_features=5, random_state=42),
            make_pipeline(StandardScaler(), LinearRegression()),
            None,
            (6, 1),
        ),
        (
            make_regression(n_features=5, n_targets=3, random_state=42),
            LinearRegression(),
            "Target",
            (6, 3),
        ),
    ],
)
def test_estimator_report_coefficients_numpy_arrays(
    data, estimator, column_base_name, expected_shape
):
    X, y = data
    estimator.fit(X, y)
    report = EstimatorReport(estimator)
    result = report.feature_importance.coefficients()
    assert result.shape == expected_shape

    expected_index = (
        ["Intercept"] + [f"x{i}" for i in range(X.shape[1])]
        if isinstance(estimator, Pipeline)
        else ["Intercept"] + [f"Feature #{i}" for i in range(X.shape[1])]
    )
    assert result.index.tolist() == expected_index

    expected_columns = (
        ["Coefficient"]
        if expected_shape[1] == 1
        else [f"{column_base_name} #{i}" for i in range(expected_shape[1])]
    )
    assert result.columns.tolist() == expected_columns


@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        make_pipeline(StandardScaler(), LinearRegression()),
    ],
)
def test_estimator_report_coefficients_pandas_dataframe(estimator):
    """If provided, the coefficients dataframe uses the feature names."""
    X, y = make_regression(n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"my_feature_{i}" for i in range(X.shape[1])])
    estimator.fit(X, y)

    report = EstimatorReport(estimator)
    result = report.feature_importance.coefficients()

    assert result.shape == (6, 1)
    assert result.index.tolist() == [
        "Intercept",
        "my_feature_0",
        "my_feature_1",
        "my_feature_2",
        "my_feature_3",
        "my_feature_4",
    ]
    assert result.columns.tolist() == ["Coefficient"]


@pytest.mark.parametrize(
    "estimator",
    [
        pytest.param(sklearn.svm.NuSVC(kernel="linear"), id="NuSVC"),
        pytest.param(sklearn.svm.NuSVR(kernel="linear"), id="NuSVR"),
        # pytest.param(sklearn.svm.OneClassSVM(), id="OneClassSVM"),
        pytest.param(sklearn.svm.SVC(kernel="linear"), id="SVC"),
        pytest.param(sklearn.svm.SVR(kernel="linear"), id="SVR"),
        pytest.param(sklearn.svm.LinearSVC(), id="LinearSVC"),
        pytest.param(sklearn.svm.LinearSVR(), id="LinearSVR"),
        # pytest.param(sklearn.cross_decomposition.CCA(), id="CCA"),
        # pytest.param(sklearn.cross_decomposition.PLSCanonical(), id="PLSCanonical"),
        pytest.param(sklearn.cross_decomposition.PLSRegression(), id="PLSRegression"),
        pytest.param(
            sklearn.discriminant_analysis.LinearDiscriminantAnalysis(),
            id="LinearDiscriminantAnalysis",
        ),
        # pytest.param(
        #     sklearn.compose.TransformedTargetRegressor(),
        #     id="TransformedTargetRegressor",
        # ),
        pytest.param(sklearn.linear_model.ElasticNet(), id="ElasticNet"),
        pytest.param(sklearn.linear_model.Lasso(), id="Lasso"),
        pytest.param(sklearn.linear_model.ARDRegression(), id="ARDRegression"),
        pytest.param(sklearn.linear_model.BayesianRidge(), id="BayesianRidge"),
        pytest.param(sklearn.linear_model.ElasticNet(), id="ElasticNet"),
        pytest.param(sklearn.linear_model.ElasticNetCV(), id="ElasticNetCV"),
        # pytest.param(sklearn.linear_model.GammaRegressor(), id="GammaRegressor"),
        pytest.param(sklearn.linear_model.HuberRegressor(), id="HuberRegressor"),
        pytest.param(sklearn.linear_model.Lars(), id="Lars"),
        pytest.param(sklearn.linear_model.LarsCV(), id="LarsCV"),
        pytest.param(sklearn.linear_model.Lasso(), id="Lasso"),
        pytest.param(sklearn.linear_model.LassoCV(), id="LassoCV"),
        pytest.param(sklearn.linear_model.LassoLars(), id="LassoLars"),
        pytest.param(sklearn.linear_model.LassoLarsCV(), id="LassoLarsCV"),
        pytest.param(sklearn.linear_model.LassoLarsIC(), id="LassoLarsIC"),
        pytest.param(sklearn.linear_model.LinearRegression(), id="LinearRegression"),
        pytest.param(
            sklearn.linear_model.LogisticRegression(), id="LogisticRegression"
        ),
        pytest.param(
            sklearn.linear_model.LogisticRegressionCV(), id="LogisticRegressionCV"
        ),
        # pytest.param(
        #     sklearn.linear_model.MultiTaskElasticNet(), id="MultiTaskElasticNet"
        # ),
        # pytest.param(
        #     sklearn.linear_model.MultiTaskElasticNetCV(), id="MultiTaskElasticNetCV"
        # ),
        # pytest.param(sklearn.linear_model.MultiTaskLasso(), id="MultiTaskLasso"),
        # pytest.param(sklearn.linear_model.MultiTaskLassoCV(), id="MultiTaskLassoCV"),
        pytest.param(
            sklearn.linear_model.OrthogonalMatchingPursuit(),
            id="OrthogonalMatchingPursuit",
        ),
        pytest.param(
            sklearn.linear_model.OrthogonalMatchingPursuitCV(),
            id="OrthogonalMatchingPursuitCV",
        ),
        pytest.param(
            sklearn.linear_model.PassiveAggressiveClassifier(),
            id="PassiveAggressiveClassifier",
        ),
        pytest.param(
            sklearn.linear_model.PassiveAggressiveRegressor(),
            id="PassiveAggressiveRegressor",
        ),
        pytest.param(sklearn.linear_model.Perceptron(), id="Perceptron"),
        # pytest.param(sklearn.linear_model.PoissonRegressor(), id="PoissonRegressor"),
        pytest.param(sklearn.linear_model.QuantileRegressor(), id="QuantileRegressor"),
        pytest.param(sklearn.linear_model.Ridge(), id="Ridge"),
        pytest.param(sklearn.linear_model.RidgeClassifier(), id="RidgeClassifier"),
        pytest.param(sklearn.linear_model.RidgeClassifierCV(), id="RidgeClassifierCV"),
        pytest.param(sklearn.linear_model.RidgeCV(), id="RidgeCV"),
        pytest.param(sklearn.linear_model.SGDClassifier(), id="SGDClassifier"),
        # pytest.param(sklearn.linear_model.SGDOneClassSVM(), id="SGDOneClassSVM"),
        pytest.param(sklearn.linear_model.SGDRegressor(), id="SGDRegressor"),
        pytest.param(sklearn.linear_model.TheilSenRegressor(), id="TheilSenRegressor"),
        pytest.param(sklearn.linear_model.TweedieRegressor(), id="TweedieRegressor"),
    ],
)
def test_all_sklearn_estimators(
    request, estimator, regression_data, classification_data
):
    """Check that `coefficients` is supported for every sklearn estimator."""
    if is_classifier(estimator):
        X, y = classification_data
    elif is_regressor(estimator):
        X, y = regression_data
    else:
        raise Exception("Estimator is neither a classifier nor a regressor")

    estimator.fit(X, y)

    report = EstimatorReport(estimator)
    result = report.feature_importance.coefficients()

    assert result.shape == (6, 1)
    assert result.index.tolist() == [
        "Intercept",
        "Feature #0",
        "Feature #1",
        "Feature #2",
        "Feature #3",
        "Feature #4",
    ]
    assert result.columns.tolist() == ["Coefficient"]


def test_pipeline_with_transformer(regression_data):
    """If the estimator is a pipeline containing a transformer that changes the
    features, adapt the feature names in the output table."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures

    X, y = regression_data
    X = pd.DataFrame(X, columns=[f"my_feature_{i}" for i in range(5)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = make_pipeline(
        PolynomialFeatures(degree=2, interaction_only=True),
        LinearRegression(),
    )

    report = EstimatorReport(
        model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    result = report.feature_importance.coefficients()
    assert result.shape == (17, 1)
    assert result.index.tolist() == [
        "Intercept",
        "1",
        "my_feature_0",
        "my_feature_1",
        "my_feature_2",
        "my_feature_3",
        "my_feature_4",
        "my_feature_0 my_feature_1",
        "my_feature_0 my_feature_2",
        "my_feature_0 my_feature_3",
        "my_feature_0 my_feature_4",
        "my_feature_1 my_feature_2",
        "my_feature_1 my_feature_3",
        "my_feature_1 my_feature_4",
        "my_feature_2 my_feature_3",
        "my_feature_2 my_feature_4",
        "my_feature_3 my_feature_4",
    ]
    assert result.columns.tolist() == ["Coefficient"]
