import pandas as pd
import pytest
from sklearn.base import is_classifier, is_regressor
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from skore import EstimatorReport


def test(classification_data):
    X, y = classification_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    forest = RandomForestClassifier(random_state=0)
    report = EstimatorReport(
        forest, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    result = report.feature_importance.mean_decrease_impurity()

    assert result.shape == (5, 1)
    assert result.index.tolist() == [
        "Feature #0",
        "Feature #1",
        "Feature #2",
        "Feature #3",
        "Feature #4",
    ]
    assert result.columns.tolist() == ["Mean decrease impurity"]


@pytest.mark.parametrize(
    "data, estimator, expected_shape",
    [
        (
            make_classification(n_features=5, random_state=42),
            RandomForestClassifier(random_state=0),
            (5, 1),
        ),
        (
            make_classification(n_features=5, random_state=42),
            RandomForestClassifier(random_state=0),
            (5, 1),
        ),
        (
            make_classification(
                n_features=5,
                n_classes=3,
                n_samples=30,
                n_informative=3,
                random_state=42,
            ),
            RandomForestClassifier(random_state=0),
            (5, 1),
        ),
        (
            make_classification(
                n_features=5,
                n_classes=3,
                n_samples=30,
                n_informative=3,
                random_state=42,
            ),
            make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0)),
            (5, 1),
        ),
        (
            make_classification(n_features=5, random_state=42),
            make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0)),
            (5, 1),
        ),
        (
            make_regression(n_features=5, n_targets=3, random_state=42),
            RandomForestRegressor(random_state=0),
            (5, 1),
        ),
    ],
)
def test_numpy_arrays(data, estimator, expected_shape):
    X, y = data
    estimator.fit(X, y)
    report = EstimatorReport(estimator)
    result = report.feature_importance.mean_decrease_impurity()

    assert result.shape == expected_shape

    expected_index = (
        [f"x{i}" for i in range(X.shape[1])]
        if isinstance(estimator, Pipeline)
        else [f"Feature #{i}" for i in range(X.shape[1])]
    )
    assert result.index.tolist() == expected_index

    expected_columns = ["Mean decrease impurity"]
    assert result.columns.tolist() == expected_columns


@pytest.mark.parametrize(
    "estimator",
    [
        RandomForestClassifier(random_state=0),
        make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0)),
    ],
)
def test_pandas_dataframe(estimator):
    """If provided, the `mean_decrease_impurity` dataframe uses the feature names."""
    X, y = make_classification(n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"my_feature_{i}" for i in range(X.shape[1])])
    estimator.fit(X, y)

    report = EstimatorReport(estimator)
    result = report.feature_importance.mean_decrease_impurity()

    assert result.shape == (5, 1)
    assert result.index.tolist() == [
        "my_feature_0",
        "my_feature_1",
        "my_feature_2",
        "my_feature_3",
        "my_feature_4",
    ]
    assert result.columns.tolist() == ["Mean decrease impurity"]


@pytest.mark.parametrize(
    "estimator",
    [
        # sklearn.ensemble.AdaBoostClassifier.feature_importances_ (Python property, in AdaBoostClassifier)
        # sklearn.ensemble.AdaBoostRegressor.feature_importances_ (Python property, in AdaBoostRegressor)
        # sklearn.ensemble.ExtraTreesClassifier.feature_importances_ (Python property, in ExtraTreesClassifier)
        # sklearn.ensemble.ExtraTreesRegressor.feature_importances_ (Python property, in ExtraTreesRegressor)
        # sklearn.ensemble.GradientBoostingClassifier.feature_importances_ (Python property, in GradientBoostingClassifier)
        # sklearn.ensemble.GradientBoostingRegressor.feature_importances_ (Python property, in GradientBoostingRegressor)
        # sklearn.ensemble.RandomForestClassifier.feature_importances_ (Python property, in RandomForestClassifier)
        # sklearn.ensemble.RandomForestRegressor.feature_importances_ (Python property, in RandomForestRegressor)
        # sklearn.ensemble.RandomTreesEmbedding.feature_importances_ (Python property, in RandomTreesEmbedding)
        # sklearn.tree.DecisionTreeClassifier.feature_importances_ (Python property, in DecisionTreeClassifier)
        # sklearn.tree.DecisionTreeRegressor.feature_importances_ (Python property, in DecisionTreeRegressor)
        # sklearn.tree.ExtraTreeClassifier.feature_importances_ (Python property, in ExtraTreeClassifier)
        # sklearn.tree.ExtraTreeRegressor.feature_importances_ (Python property, in ExtraTreeRegressor)
        # 1.11. Ensembles: Gradient boosting, random forests, bagging, voting, stacking
        # ...ce evaluation for more details). The feature importance scores of a fit gradient boosting model can be accessed via the feature_importances_ property: >>> from sklearn.datasets import make_hastie_10_2 >>> from sklearn.ensemble import Gradie...
        # 1.13. Feature selection
        # ...al set of features and the importance of each feature is obtained either through any specific attribute (such as coef_, feature_importances_) or callable. Then, the least important features are pruned from current set of features. That proc...
        # AdaBoostClassifier
        # ...boosted ensemble. estimator_errors_ndarray of floatsClassification error for each estimator in the boosted ensemble. feature_importances_ndarray of shape (n_features,)The impurity-based feature importances. n_features_in_intNumber of fe...
        # AdaBoostRegressor
        # ...the boosted ensemble. estimator_errors_ndarray of floatsRegression error for each estimator in the boosted ensemble. feature_importances_ndarray of shape (n_features,)The impurity-based feature importances. n_features_in_intNumber of fe...
        # DecisionTreeClassifier
        # ...list of ndarrayThe classes labels (single output problem), or a list of arrays of class labels (multi-output problem). feature_importances_ndarray of shape (n_features,)Return the feature importances. max_features_intThe inferred value of...
        # DecisionTreeRegressor
        # ...egressions trained on data with missing values. Read more in the User Guide. Added in version 1.4. Attributes: feature_importances_ndarray of shape (n_features,)Return the feature importances. max_features_intThe inferred value of...
        # ExtraTreeClassifier
        # ...(for single output problems), or a list containing the number of classes for each output (for multi-output problems). feature_importances_ndarray of shape (n_features,)Return the feature importances. n_features_in_intNumber of features s...
        # ExtraTreeRegressor
        # ...Names of features seen during fit. Defined only when X has feature names that are all strings. Added in version 1.0. feature_importances_ndarray of shape (n_features,)Return the feature importances. n_outputs_intThe number of outputs wh...
        # ExtraTreesClassifier
        # ...of classes (single output problem), or a list containing the number of classes for each output (multi-output problem). feature_importances_ndarray of shape (n_features,)The impurity-based feature importances. n_features_in_intNumber of fe...
        # ExtraTreesRegressor
        # ...timator_ was renamed to estimator_. estimators_list of DecisionTreeRegressorThe collection of fitted sub-estimators. feature_importances_ndarray of shape (n_features,)The impurity-based feature importances. n_features_in_intNumber of fe...
        # Feature importances with a forest of trees
        # ...=0) Feature importance based on mean decrease in impurity Feature importances are provided by the fitted attribute feature_importances_ and they are computed as the mean and standard deviation of accumulation of the impurity decrease w...
        # Glossary of Common Terms and API Elements
        # ...used for prediction or transformation; transductive outputs such as labels_ or embedding_; or diagnostic data, such as feature_importances_. Common attributes are listed below. A public attribute may have the same name as a constructor par...
        # Gradient Boosting regression
        # ...are less predictive and the error bars of the permutation plot show that they overlap with 0. feature_importance = reg.feature_importances_ sorted_idx = np.argsort(feature_importance) pos = np.arange(sorted_idx.shape[0]) + 0.5 fig = plt.fi...
        # GradientBoostingClassifier
        # ...number of trees that are built at each iteration. For binary classifiers, this is always 1. Added in version 1.4.0. feature_importances_ndarray of shape (n_features,)The impurity-based feature importances. oob_improvement_ndarray of sh...
        # GradientBoostingRegressor
        # ...n_intThe number of trees that are built at each iteration. For regressors, this is always 1. Added in version 1.4.0. feature_importances_ndarray of shape (n_features,)The impurity-based feature importances. oob_improvement_ndarray of sh...
        # Model-based and sequential feature selection
        # ...however works with any model, while SelectFromModel requires the underlying estimator to expose a coef_ attribute or a feature_importances_ attribute. The forward SFS is faster than the backward SFS because it only needs to perform n_featu...
        # Permutation Importance vs Random Forest Feature Importance (MDI)
        # ...portance. import pandas as pd feature_names = rf[:-1].get_feature_names_out() mdi_importances = pd.Series( rf[-1].feature_importances_, index=feature_names ).sort_values(ascending=True) ax = mdi_importances.plot.barh() ax.set_title(...
        # Permutation Importance with Multicollinear or Correlated Features
        # ...uring training. import matplotlib.pyplot as plt import numpy as np import pandas as pd mdi_importances = pd.Series(clf.feature_importances_, index=X_train.columns) tree_importance_sorted_idx = np.argsort(clf.feature_importances_) fig, (ax...
        # RandomForestClassifier
        # ...feature names that are all strings. Added in version 1.0. n_outputs_intThe number of outputs when fit is performed. feature_importances_ndarray of shape (n_features,)The impurity-based feature importances. oob_score_floatScore of the t...
        # RandomForestRegressor
        # ...timator_ was renamed to estimator_. estimators_list of DecisionTreeRegressorThe collection of fitted sub-estimators. feature_importances_ndarray of shape (n_features,)The impurity-based feature importances. n_features_in_intNumber of fe...
        # RandomTreesEmbedding
        # ..._ was renamed to estimator_. estimators_list of ExtraTreeRegressor instancesThe collection of fitted sub-estimators. feature_importances_ndarray of shape (n_features,)The impurity-based feature importances. n_features_in_intNumber of fe...
        # RFE
        # ...stanceA supervised learning estimator with a fit method that provides information about feature importance (e.g. coef_, feature_importances_). n_features_to_selectint or float, default=NoneThe number of features to select. If None, half of...
        # RFECV
        # ...ator with a fit method that provides information about feature importance either through a coef_ attribute or through a feature_importances_ attribute. stepint or float, default=1If greater than or equal to 1, then step corresponds to the...
        # SelectFromModel
        # ...r is built. This can be both a fitted (if prefit is set to True) or a non-fitted estimator. The estimator should have a feature_importances_ or coef_ attribute after fitting. Otherwise, the importance_getter parameter should be used. thres...
    ],
)
def test_all_sklearn_estimators(
    request, estimator, regression_data, classification_data
):
    """Check that `mean_decrease_impurity` is supported for every sklearn estimator."""
    if is_classifier(estimator):
        X, y = classification_data
    elif is_regressor(estimator):
        X, y = regression_data
    else:
        raise Exception("Estimator is neither a classifier nor a regressor")

    estimator.fit(X, y)

    report = EstimatorReport(estimator)
    result = report.feature_importance.mean_decrease_impurity()

    assert result.shape == (6, 1)
    assert result.index.tolist() == [
        "Intercept",
        "Feature #0",
        "Feature #1",
        "Feature #2",
        "Feature #3",
        "Feature #4",
    ]
    assert result.columns.tolist() == ["Mean decrease impurity"]


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
        RandomForestRegressor(random_state=0),
    )

    report = EstimatorReport(
        model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    result = report.feature_importance.mean_decrease_impurity()
    assert result.shape == (16, 1)
    assert result.index.tolist() == [
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
    assert result.columns.tolist() == ["Mean decrease impurity"]
