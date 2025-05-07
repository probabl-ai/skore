install-skore:
	python -m pip install -e './skore[polars,test,sphinx,dev]'
	pre-commit install

lint:
	pre-commit run --all-files

test-backend:
	cd skore && pytest tests

test: lint test-backend
