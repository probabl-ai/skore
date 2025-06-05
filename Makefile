install-skore:
	python -m pip install -e './skore[test,sphinx,dev]'
	pre-commit install

install-skore-lts-cpu:
	python -m pip install -e './skore[test-lts-cpu,sphinx-lts-cpu,dev]'
	pre-commit install

lint:
	pre-commit run --all-files

test-backend:
	cd skore && pytest

test: lint test-backend
