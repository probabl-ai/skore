# skore

![lint and test](https://github.com/probabl-ai/skore/actions/workflows/lint-and-test.yml/badge.svg)

## Installation

For now, the only supported method to use skore is from source.
Follow the instructions in [CONTRIBUTING.md](/CONTRIBUTING.md#quick-start) to install dependencies and start the dashboard.

## Quick start

For a complete introductory example, see our [basic usage notebook](/notebooks/basic_usage.ipynb). The resulting skore dashboard has been exported to [this HTML file](https://gist.github.com/augustebaum/6b21dbd7f7d5a584fbf2c1956692574e): download it and open it in your browser to visualize it.

In your shell, run the following to create a project file `project.skore` in your current working directory:
```sh
python -m skore create project.skore
```

Run the following in your Python code to load the project:
```python
from skore import load

project = load("project.skore")
```

You can insert objects in and read objects from the project using the following commands:
```python
project.put("my int", 3)

project.get("my int")

# Strings are assumed to be Markdown:
project.put("my string", "Hello world!")

# `put` overwrites previous data
project.put("my string", "Hello again!")

project.list_keys()

project.delete_item("my int")
```

Then, in your project root (i.e. where `project.skore` is), run the following command to start the frontend locally:
```sh
python -m skore launch project.skore
```

This should automatically open a browser tab pointing at the app URL.


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
