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

## Help for common issues

### `make build-frontend` doesn't work!

Please check that your version of node is at least 20 using the following command: `node -v`
