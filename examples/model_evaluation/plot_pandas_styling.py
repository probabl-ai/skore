"""
.. _example_pandas styling:

==========================================
Applying some pandas styling to dataframes
==========================================
"""

# %%
# Classification task
# ===================

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from skore import EstimatorReport

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
rf = RandomForestClassifier(random_state=0)
rf_report = EstimatorReport(
    rf, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

# %%
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=0)
gb_report = EstimatorReport(
    gb, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

# %%
from skore import ComparisonReport

comparator = ComparisonReport(reports=[rf_report, gb_report])

# %%
df = comparator.metrics.report_metrics(pos_label=1, indicator_favorability=False)
df


# %%
def highlight_best(row):
    # For most metrics, higher is better, but for Brier score, Fit time, and Predict time, lower is better
    reverse_metrics = ["Brier score", "Fit time (s)", "Predict time (s)"]
    is_max = (
        row.index == row.idxmax()
        if row.name not in reverse_metrics
        else row.index == row.idxmin()
    )
    return ["font-weight: bold" if v else "" for v in is_max]


# Apply styling
styled_df = (
    df.style
    # Apply gradient background (higher values are better, except for specific metrics)
    .background_gradient(cmap="Blues", axis=1)
    # Bold the best value in each row
    .apply(highlight_best, axis=1)
    # Format numbers to 6 decimal places for consistency
    .format("{:.6f}")
    # Optional: adjust text alignment
    .set_properties(**{"text-align": "center"})
)

# Display the styled DataFrame
styled_df
# %%
