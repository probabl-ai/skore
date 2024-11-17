"""
==========================================================================
Using a reporting class with accessors to display cross-validation results
==========================================================================

Just an example.
"""

# %%
import tempfile
from pathlib import Path

temp_dir = tempfile.TemporaryDirectory(prefix="skore_example_")
temp_dir_path = Path(temp_dir.name)

# %%
import subprocess

subprocess.run(f"python3 -m skore create project --working-dir {temp_dir.name}".split())

# %%
import skore

project = skore.load(temp_dir_path / "project.skore")

# %%
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from skore.sklearn import CrossValidationReporter

X, y = datasets.make_classification(
    n_samples=1_000,
    n_features=20,
    class_sep=0.5,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=42,
)
classifier = linear_model.LogisticRegression(max_iter=1_000)
cv_results = cross_validate(
    classifier, X, y, return_estimator=True, return_indices=True
)
reporter = CrossValidationReporter(cv_results, X, y)

# %%
reporter.help()

# %%
fig = reporter.plot.roc(backend="plotly")
fig

# %%
fig = reporter.plot.roc(backend="matplotlib")

# %%
# I probably would like to do the following:
# project.put("reporter", reporter)

# %%
import joblib

joblib.dump(reporter, temp_dir_path / "reporter.joblib")

# %%
reporter = joblib.load(temp_dir_path / "reporter.joblib")
fig = reporter.plot.roc(backend="plotly")
fig

# %%
reporter._cache  # stuff are still cached

# %%
reporter._hash

# %%
reporter.metrics.accuracy()

# %%
reporter.metrics.precision(average="binary")

# %%
reporter.metrics.recall(average=None)

# %%
reporter.metrics.log_loss()

# %%
reporter.metrics.roc_auc()

# %%
reporter.metrics.report_stats()

# %%
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from skore.sklearn import CrossValidationReporter

X, y = datasets.make_regression(
    n_samples=1_000,
    n_features=10,
    random_state=42,
    n_targets=1,
    noise=100,
    n_informative=5,
)
regressor = linear_model.Ridge()
cv_results = cross_validate(regressor, X, y, return_estimator=True, return_indices=True)
reporter = CrossValidationReporter(cv_results, X, y)

# %%
reporter.help()

# %%
reporter.metrics.rmse()

# %%
reporter.metrics.report_stats()

# %%
reporter.plot.model_weights()

# %%
reporter.plot.model_weights(style="violinplot", add_data_points=False)

# %%
reporter.plot.model_weights(backend="plotly")

# %%
reporter.plot.model_weights(backend="plotly", style="violinplot", add_data_points=True)

# %%
# Cleanup the project
# -------------------
#
# Remove the temporary directory:
temp_dir.cleanup()

# %%
