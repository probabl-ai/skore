# 👋 Welcome to skore

![ci](https://github.com/probabl-ai/skore/actions/workflows/ci.yml/badge.svg?event=push)
![python](https://img.shields.io/badge/python-3.11%20|%203.12-blue?style=flat&logo=python)

With `skore`, data scientists can:
1. Store objects of different types from their Python code: from python lists to `scikit-learn` fitted pipelines and `plotly` figures.
2. They can **track** and  **visualize** these stored objects on a dashboard.
3. This dashboard can be exported into a HTML file.

These are only the first features: `skore` is a work in progress and it aims to be an all-inclusive library for data scientists.
Stay tuned!

<p align="center">
    <img width="100%" src="https://github.com/sylvaincom/sylvaincom.github.io/blob/master/files/probabl/skore/2024_10_08_skore_demo.gif"/>
</p>

## ⚙️ Installation

First of all, we recommended using a [virtual environment (venv)](https://docs.python.org/3/tutorial/venv.html). You need `python>=3.9`.

Then, you can install `skore` by using `pip`:
```bash
pip install -U skore
```

You can check `skore`'s latest version on
[PyPI](https://pypi.org/project/skore/).


## 🚀 Quick start

1. From your shell, initialize a `skore` Project called `project.skore` that will be in your current working directory:
```bash
python -m skore create 'project.skore'
```
2. Then, from your Python code (in the same directory), load the project and store an integer for example:
```python
from skore import load
project = load("project.skore")
project.put("my int", 3)
```
3. Finally, from your shell (in the same directory), start the UI locally:
```bash
python -m skore launch project.skore
```
This will automatically open a browser at the UI's location:
1. Create a new `View` on the top left.
2. Then, from `Elements` section on the bottom left, you can add stored items into this view, by double-cliking on them or doing drag-and-drop.

## 👨‍💻 More examples

💡 Note that after launching the dashboard, you can keep modifying current items or store new ones from your python code, and the dashboard will automatically be refreshed.

Store a `pandas` dataframe:
```python
import numpy as np
import pandas as pd

my_df = pd.DataFrame(np.random.randn(3, 3))
project.put("my_df", my_df)
```

Store a `matplotlib` figure:
```python
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]
fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
ax.plot(x)
project.put("my_figure", fig)
```

Store a `scikit-learn` fitted pipeline:
```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

diabetes = load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
my_pipeline = Pipeline(
    [("standard_scaler", StandardScaler()), ("lasso", Lasso(alpha=2))]
)
my_pipeline.fit(X, y)
project.put("my_fitted_pipeline", my_pipeline)
```

👨‍🏫 For a complete introductory example, see our [basic usage notebook](https://github.com/probabl-ai/skore/blob/main/examples/01_basic_usage.ipynb).
It shows you how to store all types of items: python lists and dictionaries, `numpy` arrays, `scikit-learn` fitted models, `matplotlib`, `altair`, and `plotly` figures, etc.
The resulting `skore` report has been exported to [this HTML file](https://sylvaincom.github.io/files/probabl/skore/basic_usage.html).

## 🔨 Contributing

Thank you for your interest!
See [CONTRIBUTING.md](https://github.com/probabl-ai/skore/blob/main/CONTRIBUTING.md).

## 💬 Where to ask questions

| Type                                | Platforms                        |
|-------------------------------------|----------------------------------|
| 🐛 Bug reports                  | [GitHub Issue Tracker]           |
| ✨ Feature requests and ideas      | [GitHub Issue Tracker] & [Discord] |
| 💬 Usage questions, discussions, contributions, etc              | [Discord]   |

[GitHub Issue Tracker]: https://github.com/probabl-ai/skore/issues
[Discord]: https://discord.gg/scBZerAGwW

---

Brought to you by:

<a href="https://probabl.ai" target="_blank">
    <img width="120" src="https://sylvaincom.github.io/files/probabl/logo_probabl.svg" alt="Probabl logo">
</a>
