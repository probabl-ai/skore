# %% [markdown]
#
# Generate a data for a dataframe that could be the available metadata for a given
# experiment.

# %%
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

known_ml_tasks = ["binary-classification", "multiclass-classification", "regression"]
ml_task = rng.choice(known_ml_tasks, size=10)

index_reg_vs_clf = rng.choice([True, False], size=10)

r2_score = rng.uniform(0, 1, size=10)

r2_score[index_reg_vs_clf] = np.nan

accuracy_score = rng.uniform(0, 1, size=10)
precision_score = rng.uniform(0, 1, size=10)
recall_score = rng.uniform(0, 1, size=10)

accuracy_score[~index_reg_vs_clf] = np.nan
precision_score[~index_reg_vs_clf] = np.nan
recall_score[~index_reg_vs_clf] = np.nan


df = pd.DataFrame(
    {
        "ml_task": ml_task,
        "r2_score": r2_score,
        "accuracy_score": accuracy_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
    }
)

# %%
df


# %%
