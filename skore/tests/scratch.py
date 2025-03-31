# %% [markdown]
#
# Generate a data for a dataframe that could be the available metadata for a given
# experiment.

# %%
import numpy as np

rng = np.random.default_rng(42)

known_ml_tasks = ["binary-classification", "multiclass-classification", "regression"]
ml_task = rng.choice(known_ml_tasks, size=10)

# %%
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
estimator = LogisticRegression().fit(X_train, y_train)
from skore import EstimatorReport

report = EstimatorReport(estimator, X_test=X_test, y_test=y_test)

# %%
report.metrics.report_metrics(scoring_kwargs={"average": "macro"})

# %%
