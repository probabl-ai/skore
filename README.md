# 👋 Welcome to skore

![lint](https://github.com/probabl-ai/skore/actions/workflows/lint.yml/badge.svg)
![tests](https://github.com/probabl-ai/skore/actions/workflows/backend.yml/badge.svg)
![UI tests](https://github.com/probabl-ai/skore/actions/workflows/skore-ui.yml/badge.svg)

`skore` allows data scientists to create beautiful reports from their Python code:
1. Users can store objects of different types (python lists and dictionaries, `numpy` arrays, `scikit-learn` fitted models, `matplotlib` graphs, etc). Storing some values over time allows one to perform **tracking** and also to **visualize** them:
2. They can visualize these stored objects on a dashboard. The dashboard is user-friendly: objects can easily be organized.
3. This dashboard can be exported, for example into a HTML file.

These are only the first features of `skore`'s roadmap.
`skore` is a work in progress and, on the long run, it aims to be an all-inclusive library for data scientists.
Stay tuned!

## ⚙️ Installation

You can install `skore` by using `pip`:
```bash
pip install -U skore
```

## 🚀 Quick start

In your shell, run the following to create a project file `project.skore` (the default) in your current working directory:
```bash
python -m skore create
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
print(project.list_keys())

# delete an item
project.delete_item("my int")
```

Then, in the directory containing your project, run the following command in your shell to start the UI locally:
```bash
python -m skore launch project.skore
```
This will automatically open a browser at the UI's location.
You can then drag and drop items into the dashboard and start editing your report.

💡 Note that after launching the dashboard, you can keep modifying current items or store new ones, and the dashboard will automatically be refreshed.

👨‍🏫 For a complete introductory example, see our [basic usage notebook](/examples/basic_usage.ipynb).
It shows you how to store all types of items: python lists and dictionaries, `numpy` arrays, `scikit-learn` fitted models, `matplotlib` graphs, etc.
The resulting `skore` report has been exported to [this HTML file](https://gist.github.com/augustebaum/6b21dbd7f7d5a584fbf2c1956692574e): download it and open it in your favorite browser to visualize it.

## 🔨 Contributing

Thank you for your interest!
See [CONTRIBUTING.md](/CONTRIBUTING.md).

## 💬 Where to ask questions

| Type                                | Platforms                        |
|-------------------------------------|----------------------------------|
| 🐛 **Bug Reports**                  | [GitHub Issue Tracker]           |
| ✨ **Feature Requests & Ideas**      | [GitHub Issue Tracker] & [Discord] |
| 💻 **Usage Questions**              | [Discord]   |
| 💬 **General Discussion**           | [Discord]   |
| 🔨 **Contribution & Development**   | [Discord]                          |

[GitHub Issue Tracker]: https://github.com/probabl-ai/skore/issues
[Discord]: https://discord.gg/cUH3UhFD
