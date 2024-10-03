# Contributing guide

## Bugs and feature requests

Both bugs and feature requests can be filed in the Github issue tracker, as well as general questions and feedback.

Bug reports are welcome, especially those reported with [short, self-contained, correct examples](http://sscce.org/).

## Development

### Quick start

You'll need Python>=3.12 to build the backend and Node>=20 to build the frontend. Then, you can install dependencies and run the UI with:
```sh
make install
make build-frontend
make serve-ui
```

You are now all setup to run the library locally.
If you want to contribute, please continue with the three other sections.

### Backend

Install backend dependencies with
```sh
make install
```

You can run the API server with
```sh
make serve-api
```

When dependencies are changed in `pyproject.toml` the lockfiles should be updated via [`pip-compile`](https://github.com/jazzband/pip-tools):
```sh
make pip-compile
```

### Frontend

Install frontend dependencies with
```sh
npm install
```
in the `frontend` directory.

Run the frontend in dev mode (for hot-reloading) with
```sh
npm run dev
```
in the `frontend` directory


Then, to use the frontend
```sh
make build-frontend
make serve-ui
```

### Documentation

To build the docs:
```sh
make build-doc
```

Then, you can access the local build via:
```sh
open doc/_build/html/index.html
```

### PR format

We use the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/#summary) format, and we automatically check that the PR title fits this format.
In particular, commits are "sentence case", meaning "fix: Fix issue" passes, while "fix: fix issue" doesn't.
Our custom set of rules is in [commitlint.config.js](./commitlint.config.js).

Generally the description of a commit should start with a verb in the imperative voice, so that it would properly complete the sentence: "When applied, this commit will [...]".

### Contributing documentation

When writing documentation, whether it be online, docstrings or help messages in the CLI and in the UI, we strive to follow some conventions that are listed below. These might be updated as time goes on.

1. Argument descriptions should be written so that the following sentence makes sense: `Argument <argument> designates <argument description>`
  1. Argument descriptions start with lower case, and do not end with a period or other punctuation
  2. Argument descriptions start with "the" where relevant, and "whether" for booleans
2. Text is written in US english ("visualize" rather than "visualise")
3. In the CLI, positional arguments are written in snake case (`snake_case`), keyword arguments in kebab case (`kebab-case`)
4. When there is a default argument, it should be shown in the help message, typically with `(default: <default value>)` at the end of the message

## Help for common issues

### `make build-frontend` doesn't work!

Please check that your version of node is at least 20 using the following command: `node -v`
