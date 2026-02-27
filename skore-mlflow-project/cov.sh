#!/usr/bin/env bash
set -euo pipefail

mkdir -p coverage

JOBLIB_MULTIPROCESSING=0 python -m pytest src/ tests/ \
    --junitxml=coverage/pytest.xml \
    --cov=skore_mlflow_project \
    --cov-config=/dev/null \
    --cov-report=term-missing \
    --cov-report=xml:coverage/pytest-coverage.xml \
    --cov-report=html:coverage/htmlcov
