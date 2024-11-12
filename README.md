# 👋 Welcome to `skore`

![ci](https://github.com/probabl-ai/skore/actions/workflows/ci.yml/badge.svg?event=push)
![python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue?style=flat&logo=python)
[![pypi](https://img.shields.io/pypi/v/skore)](https://pypi.org/project/skore/)
![license](https://img.shields.io/pypi/l/skore)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.probabl.ai/)

With skore, data scientists can:
1. Track and visualize their ML/DS results.
2. Get assistance when developing their ML/DS projects.
    - Scikit-learn compatible `cross_validate()` provides insights and checks on cross-validation.

These are only the first features: skore is a work in progress and aims to be an end-to-end library for data scientists.
Stay tuned, and join [our Discord](https://discord.probabl.ai) if you want to give us feedback!

![GIF: short demo of skore](https://raw.githubusercontent.com/sylvaincom/sylvaincom.github.io/master/files/probabl/skore/2024_10_31_skore_demo_compressed.gif)

## ⚙️ Installation

First of all, we recommend using a [virtual environment (venv)](https://docs.python.org/3/tutorial/venv.html). You need `python>=3.9`.

Then, you can install skore by using `pip`:
```bash
pip install -U skore
```

**Warning:** For Windows users, the encoding must be set to [UTF-8](https://docs.python.org/3/using/windows.html#utf-8-mode): see [PYTHONUTF8](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONUTF8).

## 🚀 Quick start

(For more information on how and why to use skore, see our [documentation](https://probabl-ai.github.io/skore/latest/auto_examples/index.html).)

1. From your shell, initialize a skore project, here named `my_project`:
```bash
skore create "my_project"
```
This will create a skore project directory named `my_project.skore` in your current working directory.

2. Now that the project file exists, from your Python code (in the same directory), load the project so that you can read from and write to it, for example you can store an integer:
```python
from skore import load
project = load("my_project.skore")
project.put("my_int", 3)
```

3. Finally, from your shell (in the same directory), start the UI locally:
```bash
skore launch "my_project"
```
This will automatically open a browser at the UI's location:
1. On the top left, by default, you can observe that you are in a _View_ called `default`. You can rename this view or create another one.
2. From the _Items_ section on the bottom left, you can add stored items to this view, either by clicking on `+` or by doing drag-and-drop.
3. In the skore UI on the right, you can drag-and-drop items to re-order them, remove items, etc.

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
