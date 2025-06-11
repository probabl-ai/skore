"""
.. _example_dataset_display:

=======================================================
Pretty plots of your dataset with a single line of code
=======================================================

We :ref:`previously explored<_example_estimator_report>` how
:class:`skore.EstimatorReport` can accelerate your ML analysis by bringing a lot
of scikit-learn techniques under the same roof.

Let's now take a look at the ``EstimatorReport.data`` accessor and see how to make
simple but efficient plots of your dataset.

We begin by loading a dataset whose task is to predict employee salaries, and get
a baseline pipeline using :func:`skrub.tabular_learner`.
"""

# %%
import skrub
from skrub.datasets import fetch_employee_salaries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

data = fetch_employee_salaries()
X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, random_state=0)

pipeline = skrub.tabular_learner(GradientBoostingRegressor())
pipeline

# %%
# We bring the dataset and pipeline into the report, and use `.data.analyze()` to get
# our insights. The display direct representation is a :class:`skrub.TableReport` of
# the input dataset.
from skore import EstimatorReport

report = EstimatorReport(
    pipeline,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
display = report.data.analyze(dataset="all", with_y=True)
display

# %%
display.help()

# %%
display.plot_dist(x_col="gender")
# %%

display.plot_dist(x_col="gender", c_col="current_annual_salary")
