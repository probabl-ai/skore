"""
.. _example_dataset_display:

=======================================================
Pretty plots of your dataset with a single line of code
=======================================================

We :ref:`previously explored<example_estimator_report>` how
:class:`skore.EstimatorReport` can accelerate your ML analysis by bringing a lot
of scikit-learn techniques under the same roof.

Let's now take a look at the ``EstimatorReport.data`` accessor and see how to make
simple but efficient plots of your dataset.

We begin by loading a dataset whose task is to predict employee salaries, and get
a baseline pipeline using ``skrub.tabular_learner``.
"""

# %%
import skrub
from skrub.datasets import fetch_employee_salaries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

data = fetch_employee_salaries()
X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, random_state=0)

pipeline = skrub.tabular_learner(HistGradientBoostingRegressor())
pipeline

# %%
# We bring the dataset and pipeline into the report, and use ``.data.analyze()`` to get
# our insights. ``dataset="all"`` means analyzing both train and test, and ``with_y=True``
# include the target in the analysis.
#
# The direct representation of the display is a :class:`skrub.TableReport`
# of the input dataset.
from skore import EstimatorReport

report = EstimatorReport(
    pipeline,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
display = report.data.analyze(source_dataset="all", with_y=True)
display

# %%
# As usual, we can easy glance at the options using ``.help``:
display.help()

# %%
# To begin with, let's plot the gender distribution to check whether we have some
# population bias:
display.plot(x="gender")
# %%
# The dataset is somewhat balanced, with a clear majority of males.
# Next, we colorize this distribution using the salary to predict, in the column
# ``current_annual_salary``.

display.plot(x="gender", hue="current_annual_salary")

# %%
# Interestingly, we see that the median (the black vertical bar) is slightly higher for
# males. We can also see there is an outlier at $300,000.
#
# Let's now add a third dimension to this plot by visualizing the hired year as the
# y-axis (which becomes the x-axis since the plot is horizontal):
display.plot(x="gender", y="year_first_hired", hue="current_annual_salary")

# %%
# The year has replaced the salary as the x-axis, and the salary is still represented by
# the color. This plot is getting a bit hard to read due to the large number of data
# points, we can subsample it slightly to see a pattern emerges:
report.data.analyze(source_dataset="all", with_y=True, subsample=1000).plot(
    x="gender",
    y="year_first_hired",
    hue="current_annual_salary",
)
# %%
# As expected, newcomers (at the right of the plot) are paid significantly less than
# more senior employees (at the left of the plot). Let's now switch gears and observe
# the correlation among our columns:
#
# Since Pearson's correlation is only defined between numerical columns and our
# dataset contains mostly categorical columns, we're missing associations between
# most of the columns.
#
# To get a broader view of our columns correlations, we can use another metric instead,
# the `Cramer's V correlation <https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V>`_,
# whose interpretation is close to the Pearson's correlation:
#
# Let's also tweak the keyword arguments of the heatmap to change the color map.
display.plot(kind="corr", heatmap_kwargs={"cmap": "viridis"})
# %%
