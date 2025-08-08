import pytest
import sklearn
import sklearn.cross_decomposition
from sklearn.base import is_classifier, is_regressor
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from skore import CrossValidationReport
from skore._externals._sklearn_compat import get_tags


@pytest.mark.parametrize(
    "data, estimator, expected_shape",
    [
        (
            make_regression(n_features=5, random_state=42),
            LinearRegression(),
            (5, 6),
        ),
        (
            make_classification(n_features=10, random_state=42),
            LogisticRegression(),
            (5, 11),
        ),
        (
            make_regression(n_features=5, random_state=42),
            make_pipeline(StandardScaler(), LinearRegression()),
            (5, 6),
        ),
    ],
)
def test_cross_validation_report_coefficient_frame(
    data,
    estimator,
    expected_shape,
):
    X, y = data
    cv_report = CrossValidationReport(estimator, X=X, y=y, splitter=5)
    cv_report_coefs = cv_report.feature_importance.coefficients().frame()
    assert cv_report_coefs.shape == expected_shape

    expected_index = [i for i in range(expected_shape[0])]
    assert cv_report_coefs.index.tolist() == expected_index

    expected_columns = (
        ["Intercept"] + [f"x{i}" for i in range(X.shape[1])]
        if isinstance(estimator, Pipeline)
        else ["Intercept"] + [f"Feature #{i}" for i in range(X.shape[1])]
    )
    assert cv_report_coefs.columns.tolist() == expected_columns


@pytest.mark.parametrize(
    "estimator",
    [
        pytest.param(sklearn.svm.NuSVC(kernel="linear"), id="NuSVC"),
        pytest.param(sklearn.svm.NuSVR(kernel="linear"), id="NuSVR"),
        pytest.param(sklearn.svm.SVC(kernel="linear"), id="SVC"),
        pytest.param(sklearn.svm.SVR(kernel="linear"), id="SVR"),
        pytest.param(sklearn.svm.LinearSVC(), id="LinearSVC"),
        pytest.param(sklearn.svm.LinearSVR(), id="LinearSVR"),
        pytest.param(sklearn.cross_decomposition.PLSRegression(), id="PLSRegression"),
        pytest.param(
            sklearn.discriminant_analysis.LinearDiscriminantAnalysis(),
            id="LinearDiscriminantAnalysis",
        ),
        pytest.param(
            sklearn.compose.TransformedTargetRegressor(),
            id="TransformedTargetRegressor",
        ),
        pytest.param(sklearn.linear_model.ElasticNet(), id="ElasticNet"),
        pytest.param(sklearn.linear_model.ARDRegression(), id="ARDRegression"),
        pytest.param(sklearn.linear_model.BayesianRidge(), id="BayesianRidge"),
        pytest.param(sklearn.linear_model.ElasticNet(), id="ElasticNet"),
        pytest.param(sklearn.linear_model.ElasticNetCV(), id="ElasticNetCV"),
        pytest.param(sklearn.linear_model.GammaRegressor(), id="GammaRegressor"),
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
        pytest.param(sklearn.linear_model.PoissonRegressor(), id="PoissonRegressor"),
        pytest.param(sklearn.linear_model.QuantileRegressor(), id="QuantileRegressor"),
        pytest.param(sklearn.linear_model.Ridge(), id="Ridge"),
        pytest.param(sklearn.linear_model.RidgeClassifier(), id="RidgeClassifier"),
        pytest.param(sklearn.linear_model.RidgeClassifierCV(), id="RidgeClassifierCV"),
        pytest.param(sklearn.linear_model.RidgeCV(), id="RidgeCV"),
        pytest.param(sklearn.linear_model.SGDClassifier(), id="SGDClassifier"),
        pytest.param(sklearn.linear_model.SGDRegressor(), id="SGDRegressor"),
        pytest.param(sklearn.linear_model.TheilSenRegressor(), id="TheilSenRegressor"),
        pytest.param(sklearn.linear_model.TweedieRegressor(), id="TweedieRegressor"),
        # The following models would be tested in the future when the `EstimatorReport`
        # will have metrics specific to these models:
        #
        # 1. multi-task
        # pytest.param(
        #     sklearn.linear_model.MultiTaskElasticNet(), id="MultiTaskElasticNet"
        # ),
        # pytest.param(
        #     sklearn.linear_model.MultiTaskElasticNetCV(), id="MultiTaskElasticNetCV"
        # ),
        # pytest.param(sklearn.linear_model.MultiTaskLasso(), id="MultiTaskLasso"),
        # pytest.param(sklearn.linear_model.MultiTaskLassoCV(), id="MultiTaskLassoCV"),
        # 2. cross_decomposition
        # pytest.param(sklearn.cross_decomposition.CCA(), id="CCA"),
        # pytest.param(sklearn.cross_decomposition.PLSCanonical(), id="PLSCanonical"),
        # 3. outlier detectors
        # pytest.param(sklearn.linear_model.SGDOneClassSVM(), id="SGDOneClassSVM"),
        # pytest.param(sklearn.svm.OneClassSVM(kernel="linear"), id="OneClassSVM"),
    ],
)
def test_all_sklearn_estimators(
    request,
    estimator,
    regression_data,
    positive_regression_data,
    binary_classification_data,
):
    """Check that `coefficients` is supported for every sklearn estimator."""
    if is_classifier(estimator):
        X, y = binary_classification_data
    elif is_regressor(estimator):
        if get_tags(estimator).target_tags.positive_only:
            X, y = positive_regression_data
        else:
            X, y = regression_data
    else:
        raise Exception("Estimator not in ['classifier', 'regressor']")

    expected_shape = (5, 6)
    cv_report = CrossValidationReport(estimator, X=X, y=y)
    cv_report_coefs = cv_report.feature_importance.coefficients().frame()

    expected_index = [i for i in range(expected_shape[0])]
    assert cv_report_coefs.index.tolist() == expected_index

    expected_columns = ["Intercept"] + [f"Feature #{i}" for i in range(X.shape[1])]
    assert cv_report_coefs.columns.tolist() == expected_columns
