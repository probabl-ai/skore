"""
==============================
Getting started with ``skore``
==============================

`This is work in progress.`

This guide provides a quick start to ``skore``, an open-source package that aims to enable data scientists to:

1. Store objects of different types from their Python code: python lists, `scikit-learn` fitted pipelines, `plotly` figures, and more.
2. Track and visualize these stored objects on a user-friendly dashboard.
3. Export the dashboard to a HTML file.

.. prompt:: bash

    twine upload dist/*
"""

# %%
# From your shell, initialize a skore project, here named project.skore, that will be in your current working directory:
# ```
# python -m skore create "project.skore"
# ```


# %%
import sklearn
import matplotlib.pyplot as plt

# %%
sklearn.__version__
plt.plot([0, 1, 2, 3], [0, 1, 2, 3])
plt.show()
