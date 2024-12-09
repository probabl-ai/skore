# 👋 Welcome to `skore`

<div style="text-align: center;">
    <img width="45%" src="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Logo_Skore_Light@2x.svg" alt="skore logo">
</div>

![ci](https://github.com/probabl-ai/skore/actions/workflows/ci.yml/badge.svg?event=push)
![python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue?style=flat&logo=python)
[![pypi](https://img.shields.io/pypi/v/skore)](https://pypi.org/project/skore/)
[![downloads](https://static.pepy.tech/badge/skore/month)](https://pepy.tech/projects/skore)
![license](https://img.shields.io/pypi/l/skore)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.probabl.ai/)

With skore, data scientists can:
1. Track and visualize their ML/DS results.
2. Get assistance when developing their ML/DS projects.
    - Scikit-learn compatible `skore.cross_validate()` and `skore.train_test_split()` provide insights and checks on cross-validation and train-test split.

These are only the first features: skore is a work in progress and aims to be an end-to-end library for data scientists.
Stay tuned! Feedbacks are welcome: please feel free to join [our Discord](https://discord.probabl.ai).

![GIF: short demo of skore](https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/2024_10_31_skore_demo_compressed.gif)

## ⚙️ Installation

First of all, we recommend using a [virtual environment (venv)](https://docs.python.org/3/tutorial/venv.html). You need `python>=3.9`.

Then, you can install skore by using `pip`:
```bash
pip install -U skore
```

## 🚀 Quick start

> **Note:** For more information on how and why to use skore, see our [documentation](https://skore.probabl.ai).

### Manipulating the skore UI

1. From your Python code, create and load a skore project, here named `my_project`:
    ```python
    import skore
    my_project = skore.create("my_project")
    ```
This will create a skore project directory named `my_project.skore` in your current working directory.

2. Start storing some items, for example you can store an integer:
    ```python
    my_project.put("my_int", 3)
    ```
    or the result of a scikit-learn grid search:
    ```python
    import numpy as np
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV

    X, y = load_diabetes(return_X_y=True)

    gs_cv = GridSearchCV(
        Ridge(),
        param_grid={"alpha": np.logspace(-3, 5, 50)},
        scoring="neg_root_mean_squared_error",
    )
    gs_cv.fit(X, y)

    my_project.put("my_gs_cv", gs_cv)
    ```

3. Finally, from your shell (in the same directory), start the UI locally:
    ```bash
    skore launch "my_project"
    ```
    This will automatically open a browser at the UI's location.

On the UI:
1. On the top menu, by default, you can observe that you are in a _View_ called `default`. You can rename this view or create another one.
2. From the _Items_ section on the left, you can add stored items to this view, either by clicking on `+` or by dragging an item to the right.
3. In the skore UI on the right, you can drag-and-drop items to re-order them, remove items, etc.

### Get assistance when developing your ML/DS projects

By using `skore.cross_validate()`:
```python
import skore
my_project = skore.create("my_project")

from sklearn.datasets import load_iris
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
clf = SVC(kernel="linear", C=1, random_state=0)

cv_results = skore.cross_validate(clf, X, y, cv=5, project=my_project)
```
You will automatically be able to visualize some key metrics (although you might have forgotten to specify all of them):
![GIF: short demo of skore](https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/2024_11_21_cross_val_comp.gif)

There is also a train-test split function that enhances scikit-learn. See more in our [documentation](https://skore.probabl.ai).

## 🔨 Contributing

Thank you for your interest!
See [CONTRIBUTING.rst](https://github.com/probabl-ai/skore/blob/main/CONTRIBUTING.rst).

## 💬 Where to ask questions

| Type                                | Platforms                        |
|-------------------------------------|----------------------------------|
| 🐛 Bug reports                  | [GitHub Issue Tracker]           |
| ✨ Feature requests and ideas      | [GitHub Issue Tracker] & [Discord] |
| 💬 Usage questions, discussions, contributions, etc              | [Discord]   |

[GitHub Issue Tracker]: https://github.com/probabl-ai/skore/issues
[Discord]: https://discord.gg/scBZerAGwW

---

Brought to you by

<a href="https://probabl.ai" target="_blank">
    <img width="120" src="https://sylvaincom.github.io/files/probabl/Logo-orange.png" alt="Probabl logo">
</a>
