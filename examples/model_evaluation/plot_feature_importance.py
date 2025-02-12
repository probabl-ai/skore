"""
.. _example_feature_importance:

==========================================================================================
`Feature Importance`: Understand which feature is the most interesting through permutation
==========================================================================================

This example shows how the :class:`skore.EstimatorReport` class can be used to
quickly get insights from any scikit-learn estimator features.
"""

# %%
# import to build the datasets
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# import to test tree models
from sklearn.tree import DecisionTreeRegressor

# import to test SVM
from sklearn.svm import SVC

# import to test ensemble models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# import to test others
from sklearn.linear_model import LogisticRegression

# %%
X_reg_bin, y_reg_bin = make_regression(
    n_samples=1000, n_features=10, n_informative=5, noise=0.1
)
X_reg_multi, y_reg_multi = make_regression(
    n_samples=1000, n_features=10, n_informative=5, noise=0.1, n_targets=3
)

X_cl_bin, y_cl_bin = make_classification(
    n_samples=1000, n_features=10, n_informative=5, n_classes=2
)
X_cl_multi, y_cl_multi = make_classification(
    n_samples=1000, n_features=10, n_informative=5, n_classes=3
)

X_train_reg_bin, X_test_reg_bin, y_train_reg_bin, y_test_reg_bin = train_test_split(
    X_reg_bin, y_reg_bin
)
X_train_reg_multi, X_test_reg_multi, y_train_reg_multi, y_test_reg_multi = (
    train_test_split(X_reg_multi, y_reg_multi)
)
X_train_cl_bin, X_test_cl_bin, y_train_cl_bin, y_test_cl_bin = train_test_split(
    X_cl_bin, y_cl_bin
)
X_train_cl_multi, X_test_cl_multi, y_train_cl_multi, y_test_cl_multi = train_test_split(
    X_cl_multi, y_cl_multi
)

# %%
# from skore import EstimatorReport
# clf = KernelRigde.fit(X_train_reg_bin, y_train_reg_bin)
# EstimatorReport(estimator = clf, X_train = X_train_reg_bin, y_train = y_train_reg_bin, X_test = X_test_reg_bin, y_test = y_test_reg_bin).plots.feature_importance(scoring = "scoring")

# %%
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas as pd


def permute(clf, X_train, y_train, X_test_, y_test, scoring=None):
    clf.fit(X_train, y_train)
    feature_names = [f"feature {i}" for i in range(X_train.shape[1])]
    result = permutation_importance(
        clf, X_test_, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring=scoring
    )
    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    # sort the Series forest_importances by decreasing value
    forest_importances = forest_importances.sort_values(ascending=False)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on the test set")
    if scoring is None:
        scoring = "default score"
    ax.set_ylabel(f"Mean {scoring} decrease")
    # must be precised to adapt decrease / increase depending on the scoring
    fig.tight_layout()
    plt.show()


# %%
## Regression binary
permute(
    DecisionTreeRegressor(),
    X_train_reg_bin,
    y_train_reg_bin,
    X_test_reg_bin,
    y_test_reg_bin,
)
# %%
permute(
    RandomForestRegressor(),
    X_train_reg_bin,
    y_train_reg_bin,
    X_test_reg_bin,
    y_test_reg_bin,
)

# %%
permute(
    RandomForestClassifier(),
    X_train_cl_bin,
    y_train_cl_bin,
    X_test_cl_bin,
    y_test_cl_bin,
)
# %%
permute(SVC(), X_train_cl_bin, y_train_cl_bin, X_test_cl_bin, y_test_cl_bin)
# %%
permute(
    LogisticRegression(),
    X_train_cl_multi,
    y_train_cl_multi,
    X_test_cl_multi,
    y_test_cl_multi,
)
# %%
