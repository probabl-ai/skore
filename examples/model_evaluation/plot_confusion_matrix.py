"""
.. _example_confusion_matrix:

===============================================================
Using Confusion Matrix in EstimatorReport
===============================================================

This example shows how to use the confusion matrix feature in :class:`skore.EstimatorReport`
to visualize classification results.
"""

# %%
# Loading our dataset and defining our estimator
# ==============================================
#
# First, we load the iris dataset, which is a multi-class classification problem.
# We'll use a simple RandomForestClassifier to demonstrate the confusion matrix.

# %%
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Create and train a classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# %%
# Creating an EstimatorReport
# ===========================
#
# Now we create an EstimatorReport for our classifier and use it to visualize
# the results.

# %%
from skore import EstimatorReport

report = EstimatorReport(
    clf, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

# %%
# Exploring available metrics
# ==========================
#
# Let's see what metrics are available for our classifier:

# %%
report.metrics.help()

# %%
# Using the confusion matrix
# =========================
#
# Now, let's visualize the confusion matrix for our classifier:

# %%
# Basic confusion matrix
report.metrics.confusion_matrix()

# %%
# We can also normalize the confusion matrix to better understand the performance
# as percentages of the true labels:

# %%
report.metrics.confusion_matrix(normalize="true")

# %%
# We can obtain the confusion matrix as a pandas DataFrame for further analysis:

# %%
cm_display = report.metrics.confusion_matrix()
cm_table = cm_display.table()
cm_table

# %%
# Using different display options
# ==============================
#
# We can customize the confusion matrix display with different options:

# %%
# Using custom display labels
report.metrics.confusion_matrix(
    display_labels=["Setosa", "Versicolor", "Virginica"], cmap="Blues"
)

# %%
# Normalizing by predicted label (columns)
report.metrics.confusion_matrix(
    display_labels=["Setosa", "Versicolor", "Virginica"],
    normalize="pred",
    cmap="Greens",
)

# %%
# This example demonstrates how the confusion matrix can provide valuable insights
# into the classifier's performance, showing where misclassifications occur between classes.
