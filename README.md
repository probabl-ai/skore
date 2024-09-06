# skore

![lint and test](https://github.com/probabl-ai/skore/actions/workflows/lint-and-test.yml/badge.svg)

## Installation

For now, the only supported method to use skore is from source.
Follow the instructions in [CONTRIBUTING.md](/CONTRIBUTING.md#quick-start) to install dependencies and start the dashboard.

## Quick start

For a complete introductory example, see our [basic usage notebook](/notebooks/basic_usage.ipynb).

Initialize and use a Store as follows:
```python
from skore import Store

store = Store("root/probabl")
```

To initialize a Store, we need to give it a root path.
A store also needs some physical storage to get and put items from/into. By default, this storage will be in a `.datamander` directory in the current working directory. This can be customized by setting the MANDR_ROOT environment variable.

```python
store.insert("my int", 3)
store.read("my int")


# Strings are assumed to be Markdown:
store.insert("my string", "Hello world!")
store.update("my string", "Hello again!")

for key, value in store.items():
    print(f"Key {key} corresponds to value {value}")
```

Then, in your project root (i.e. where `.datamander` is), run the following command to start the frontend locally:
```sh
python -m skore launch .datamander
```
This should automatically open a browser tab pointing at the app URL.

## Help for common issues


`make build-frontend` doesn't work!

Please check that the node version is above 20 thanks to the following command: `node -v`

## Roadmap


With Skore, you can:
- Store data
- Visualize data

In the future, you will be able to:
- Share visualizations of your data
- Extract insights from your data
- Get tips on how to improve your data science code

## Concepts


- A **Store** is the core concept of this project. It is a dict-like data structure that implements a CRUD interface.
- A **Storage** represents the actual data storage medium, e.g. a computer's filesystem or an S3 bucket. Every Store has one Storage.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information and to contribute to the evolution of this library.
