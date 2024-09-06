# Mandr

![lint and test](https://github.com/probabl-ai/mandr/actions/workflows/lint-and-test.yml/badge.svg)

## Installation

For now, the only supported method to use mandr is from source.
Follow the instructions in [CONTRIBUTING.md](/CONTRIBUTING.md#quick-start) to install dependencies and start the dashboard.

## Basic usage

Initialize and use a Store as follows:
```python
from mandr import Store

# To initialize a Store, we need to give it a root path. This abstract path lets you express a hierarchy between Stores (so a Store can contain Stores).
# A store also needs some physical storage to get and put items from/into.
# By default, this storage will be in a `.datamander` directory in the current working directory.
# This can be customized by setting the MANDR_ROOT environment variable.
store = Store("root/probabl")

store.insert("my int", 3)

store.read("my int")

# The store infers the type of the inserted object by default, but sometimes it
# is necessary to be explicit.
# For example, strings are assumed to be Markdown:
store.insert("my string", "<p>Hello world!</p>")

# But you can tell Mandr to interpret the input as HTML
store.update("my string", "<p>Hello world!</p>", display_type="html")

for key, value in store.items():
    print(f"Key {key} corresponds to value {value}")
```

Then, in your project root (i.e. where `.datamander` is), run the following command to start the frontend locally:
```sh
python -m mandr launch .datamander
```
This should automatically open a browser tab pointing at the app URL.

## ML-specific example

```python
from mandr import Store

store = Store("root/ml_example")

# Train an sklearn estimator and evaluate it with cross-validation
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate

diabetes = load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = Lasso()

import pandas as pd
store.insert("my_cv", pd.DataFrame(cross_validate(lasso, X, y, cv=5)))
```

## Roadmap

With Mandr, you can:
- Store data (`mandr.insert()`)
- Visualize data (`mandr.launch_dashboard()`)

In the future, you can:
- Share visualizations of your data
- Extract insights from your data
- Get tips on how to improve your data science

## Concepts

- A **Store** is the core concept of this project. It is a dict-like data structure that implements a CRUD interface.
- A **Storage** represents the actual data storage medium, e.g. a computer's filesystem or an S3 bucket. Every Store has one Storage.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.
