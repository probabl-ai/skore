"""
.. _example_train_test_split:

==========================
Enhancing train-test split
==========================

This example illustrates the motivation and the use of skore's
:func:`skore.train_test_split` to get assistance when developing your
ML/DS projects.
"""

# %%
# Creating and loading the skore project
# ======================================
#
# We start by creating a temporary directory to store our project such that we can
# easily clean it after executing this example. If you want to keep the project, you
# have to skip this section.

# %%
import tempfile
from pathlib import Path

import skore

temp_dir = tempfile.TemporaryDirectory(prefix="skore_example_")
temp_dir_path = Path(temp_dir.name)

my_project = skore.create("my_project.skore", working_dir=temp_dir_path)

# %%
# Train-test split in scikit-learn
# ================================
#
# Scikit-learn holds a function for splitting the data into a training and a testing
# sets: :func:`sklearn.model_selection.train_test_split`.


# %%
from skore import train_test_split

import numpy as np

X, y = np.arange(10).reshape((5, 2)), range(5)

# Drop-in replacement for sklearn train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
X_train
