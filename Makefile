install-skore:
	python -m pip install -e './skore[test,sphinx]'
	pre-commit install

build-skore-ui:
	# cleanup
	rm -rf skore-ui/dist
	rm -rf skore/src/skore/ui/static
	# build
	( \
		cd skore-ui; \
		npm install; \
		npm run build; \
	)
	# move
	mv skore-ui/dist/ skore/src/skore/ui/static

serve-skore-ui:
	python -m uvicorn \
		--factory skore.ui.app:create_app \
		--reload --reload-dir skore/src \
		--host 0.0.0.0 \
		--port 22140 \
		--timeout-graceful-shutdown 0

lint:
	pre-commit run --all-files

test-frontend:
	cd skore-ui && npm install
	cd skore-ui && npm run test:unit

test-backend:
	cd skore && pytest tests

test: lint test-frontend test-backend
