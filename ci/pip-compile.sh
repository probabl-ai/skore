#!/usr/bin/env bash
#
# This script compiles all the `test-requirements.txt` files, based on combinations of
# `python`, `scikit-learn` and, for MLflow compatibility testing, `mlflow` versions.
# These combinations mirror those defined in the GitHub workflows.
#
# You can pass any `uv pip compile` parameter:
#
#     $ bash pip-compile.sh --test-requirements <skore|skore-hub-project|skore-local-project|skore-mlflow-project> --upgrade
#     $ bash pip-compile.sh --sphinx-requirements --upgrade
#

usage () {
    >&2 echo "Usage:"
    >&2 echo "    $ bash pip-compile.sh --test-requirements <all|skore|skore-hub-project|skore-local-project|skore-mlflow-project> [option...]"
    >&2 echo "    $ bash pip-compile.sh --sphinx-requirements [option...]"
}

CWD=$(realpath $(dirname $0))
TMPDIR=$(mktemp -d)
COMBINATIONS=()

# Make sure that `TMPDIR` is removed on exit, whatever the signal
trap 'rm -rf ${TMPDIR}' 0

# Construct `COMBINATIONS` based on arguments
case $1 in
    "--test-requirements")
        PACKAGES=()

        case $2 in
            "all")
                PACKAGES+=("skore")
                PACKAGES+=("skore-hub-project")
                PACKAGES+=("skore-local-project")
                PACKAGES+=("skore-mlflow-project")
                ;;
            "skore"|"skore-hub-project"|"skore-local-project"|"skore-mlflow-project")
                PACKAGES+=($2)
                ;;
            *)
                >&2 echo -e "Error: Unknown PACKAGE \033[0;41m$2\033[0m"
                usage
                exit 1
                ;;
        esac

        for PACKAGE in "${PACKAGES[@]}"
        do
            COMBINATIONS+=("${PACKAGE};test;3.10;1.5")
            COMBINATIONS+=("${PACKAGE};test;3.10;1.7")
            COMBINATIONS+=("${PACKAGE};test;3.11;1.5")
            COMBINATIONS+=("${PACKAGE};test;3.11;1.8")
            COMBINATIONS+=("${PACKAGE};test;3.12;1.5")
            COMBINATIONS+=("${PACKAGE};test;3.12;1.8")
            COMBINATIONS+=("${PACKAGE};test;3.13;1.5")
            COMBINATIONS+=("${PACKAGE};test;3.13;1.6")
            COMBINATIONS+=("${PACKAGE};test;3.13;1.7")
            COMBINATIONS+=("${PACKAGE};test;3.13;1.8")

            # Create additional lockfiles for the dedicated MLflow compatibility job.
            if [[ "${PACKAGE}" == "skore-mlflow-project" ]]; then
                COMBINATIONS+=("${PACKAGE};test;3.10;1.5;2.20.4")
                COMBINATIONS+=("${PACKAGE};test;3.11;1.5;2.21.3")
                COMBINATIONS+=("${PACKAGE};test;3.12;1.5;2.22.4")
                COMBINATIONS+=("${PACKAGE};test;3.10;1.7;3.0.1")
                COMBINATIONS+=("${PACKAGE};test;3.11;1.8;3.1.4")
                COMBINATIONS+=("${PACKAGE};test;3.12;1.8;3.2.0")
                COMBINATIONS+=("${PACKAGE};test;3.10;1.5;3.3.2")
                COMBINATIONS+=("${PACKAGE};test;3.11;1.5;3.4.0")
                COMBINATIONS+=("${PACKAGE};test;3.12;1.5;3.5.1")
                COMBINATIONS+=("${PACKAGE};test;3.13;1.5;3.6.0")
                COMBINATIONS+=("${PACKAGE};test;3.13;1.6;3.7.0")
                COMBINATIONS+=("${PACKAGE};test;3.13;1.7;3.8.1")
                COMBINATIONS+=("${PACKAGE};test;3.13;1.8;3.9.0")
                COMBINATIONS+=("${PACKAGE};test;3.13;1.8;3.10.0")
            fi
        done

        unset PACKAGES
        unset PACKAGE
        shift 2
        ;;
    "--sphinx-requirements")
        COMBINATIONS+=("skore;sphinx;3.13;1.8")
        shift
        ;;
    *)
        >&2 echo -e "Error: Unknown OPTION \033[0;41m$1\033[0m"
        usage
        exit 1
        ;;
esac

set -eu

(
    counter=1

    # Move to `TMPDIR` to avoid absolute paths in requirements file
    cd "${TMPDIR}"

    for combination in "${COMBINATIONS[@]}"
    do
        IFS=";" read -r PACKAGE EXTRA PYTHON SCIKIT_LEARN MLFLOW <<< "${combination}"

        if [[ -n "${MLFLOW:-}" ]]; then
            FILEPATH="${CWD}/requirements/${PACKAGE}/python-${PYTHON}/scikit-learn-${SCIKIT_LEARN}/mlflow-${MLFLOW}/${EXTRA}-requirements.txt"
            echo "Generating ${PACKAGE} ${EXTRA}-requirements: python==${PYTHON} | scikit-learn==${SCIKIT_LEARN} | mlflow==${MLFLOW} (${counter}/${#COMBINATIONS[@]})"
        else
            FILEPATH="${CWD}/requirements/${PACKAGE}/python-${PYTHON}/scikit-learn-${SCIKIT_LEARN}/${EXTRA}-requirements.txt"
            echo "Generating ${PACKAGE} ${EXTRA}-requirements: python==${PYTHON} | scikit-learn==${SCIKIT_LEARN} (${counter}/${#COMBINATIONS[@]})"
        fi

        # Copy everything necessary to compile requirements in `TMPDIR`
        mkdir -p "${TMPDIR}/${PACKAGE}"; cp "${CWD}/../${PACKAGE}/pyproject.toml" "${TMPDIR}/${PACKAGE}"

        # Force the `scikit-learn` version by creating file overriding requirements
        echo "scikit-learn==${SCIKIT_LEARN}.*" > "${PACKAGE}/overrides.txt"
        if [[ -n "${MLFLOW:-}" ]]; then
            echo "mlflow==${MLFLOW}" >> "${PACKAGE}/overrides.txt"
        fi

        # Create the requirements file tree
        mkdir -p $(dirname "${FILEPATH}")

        # Create the `.python-version` file used by `Dependabot`
        echo "${PYTHON}" > "${CWD}/requirements/${PACKAGE}/python-${PYTHON}/.python-version"

        # Create the requirements file
        uv pip compile \
           --quiet \
           --no-strip-extras \
           --no-header \
           --extra="${EXTRA}" \
           --override "${PACKAGE}/overrides.txt" \
           --python-version "${PYTHON}" \
           --output-file "${FILEPATH}" \
           "${PACKAGE}/pyproject.toml" \
           "${@:1}"

        let counter++
    done
)
