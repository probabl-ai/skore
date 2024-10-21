"""
========================
Basic usage of ``skore``
========================

*This is work in progress.*

This example builds on top of the :ref:`getting_started` guide.
"""

# %%
from skore import load

# project = load("project.skore")

# %%
# project.put("my_int", 3)

# %%
import sklearn
import matplotlib.pyplot as plt

# %%
import numpy as np
import pandas as pd

my_df = pd.DataFrame(np.random.randn(3, 3))
# project.put("my_df", my_df)

# %%
sklearn.__version__
plt.plot([0, 1, 2, 3], [0, 1, 2, 3])
plt.show()
