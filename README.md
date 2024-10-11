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

In your shell, run the following to create a project file `project.skore` (the default) in your current working directory:
```bash
python -m skore create 'project.skore'
```

Run the following in your Python code (in the same working directory) to load the project, store some objects, delete them, etc:
```python
from skore import load

# load the project
project = load("project.skore")

# save an item you need to track in your project
project.put("my int", 3)

# get an item's value
project.get("my int")

# by default, strings are assumed to be Markdown:
project.put("my string", "Hello world!")

# `put` overwrites previous data
project.put("my string", "Hello again!")

# list all the keys in a project
print(project.list_item_keys())

# delete an item
project.delete_item("my int")
```

Then, in the directory containing your project, run the following command in your shell to start the UI locally:
```bash
python -m skore launch project.skore
```
This will automatically open a browser at the UI's location:
1. Create a new `View`.
2. Then, you can add items into this view, by double-cliking on them or doing drag-and-drop. You can visualize the stored items in the `Elements` tab on the left.

💡 Note that after launching the dashboard, you can keep modifying current items or store new ones, and the dashboard will automatically be refreshed.

👨‍🏫 For a complete introductory example, see our [basic usage notebook](https://github.com/probabl-ai/skore/blob/main/examples/basic_usage.ipynb).
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
