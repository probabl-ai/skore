# skore

![lint and test](https://github.com/probabl-ai/skore/actions/workflows/lint-and-test.yml/badge.svg)

Given to you by [:probabl.](https://probabl.ai/), `skore` is a powerful tool that allows data scientists to create tracking and clear reports from their Python code, typically a notebook.

## Installation

For now, the only supported method to use skore is from source.
Follow the instructions in [CONTRIBUTING.md](/CONTRIBUTING.md#quick-start) to install dependencies and start the UI.

## Quick start

For a complete introductory example, see our [basic usage notebook](/examples/basic_usage.ipynb). The resulting skore report has been exported to [this HTML file](https://gist.github.com/augustebaum/6b21dbd7f7d5a584fbf2c1956692574e): download it and open it in your browser to visualize it.

In your shell, run the following to create a project file `project.skore` (the default) in your current working directory:
```sh
python -m skore create
```

Run the following in your Python code to load the project:
```python
from skore import load

project = load("project.skore")
```

You can save items you need to track in your project:
```python
project.put("my int", 3)
```

You can also get them back:
```python
project.get("my int")
```

By default, strings are assumed to be Markdown:
```python
project.put("my string", "Hello world!")
```

Note that `put` overwrites previous data
```python
project.put("my string", "Hello again!")
```

You can list all the keys in a project with:
```python
project.list_keys()
```

You can delete items with:
```python
project.delete_item("my int")
```

Then, in the directory containing your project, run the following command to start the UI locally:
```sh
python -m skore launch project.skore
```

This will automatically open a browser at the UI's location.


## Roadmap

With Skore, you can:
- Store data
- Visualize data

In the future, you will be able to:
- Share visualizations of your data
- Extract insights from your data
- Get tips on how to improve your data science code

## Contributing

See [CONTRIBUTING.md](/CONTRIBUTING.md) for more information and to contribute to the evolution of this library.
