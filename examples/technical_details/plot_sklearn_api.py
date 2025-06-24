"""
.. _example_sklearn_api:

===========================================================
Using skore for models compatible with the scikit-learn API
===========================================================
"""

# %%
if False:
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=500,
        n_classes=2,
        class_sep=0.3,
        n_clusters_per_class=1,
        random_state=42,
    )

# %%
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")
X.head()

# %%
from skore import train_test_split

split_data = train_test_split(X, y, random_state=42, as_dict=True)

# %%
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from skrub import tabular_learner

estimators = [
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=10_000)),
    tabular_learner("classification"),
    TabICLClassifier(),
    TabPFNClassifier(),
]

# %%
from skore import EstimatorReport

estimator_reports = [
    EstimatorReport(est, pos_label=1, **split_data) for est in estimators
]

# %%
from skore import ComparisonReport

comparator = ComparisonReport(estimator_reports)

# %%
comparator.metrics.summarize().frame()

# %%
comparator.metrics.roc().plot()
# %%
