#!/usr/bin/env bash
#
# This script compiles all the `test-requirements.txt` files, based on combinations of
# `python` and `scikit-learn` versions. These combinations mirror those defined in the
# GitHub workflows.
#
# You can pass any `uv pip compile` parameter:
#
#     $ bash pip-compile.sh --test-requirements <skore|skore-hub-project|skore-mlflow-project> --upgrade
#     $ bash pip-compile.sh --sphinx-requirements --upgrade
#

usage () {
    >&2 echo "Usage:"
    >&2 echo "    $ bash pip-compile.sh --test-requirements <all|skore|skore-hub-project|skore-mlflow-project> [option...]"
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
                PACKAGES+=("skore-mlflow-project")
                ;;
            "skore"|"skore-hub-project"|"skore-mlflow-project")
                PACKAGES+=($2)
                ;;
            *)
                >&2 echo -e "Error: Unknown PACKAGE \033[0;41m$2\033[0m"
                usage
                exit 1
                ;;
        esac

        for PACKAGE in "${PACKAGES[@]}"; do
            while IFS= read -r combination; do
                python=$(jq -rc '.python' <<< "${combination}")
                dependencies=$(jq -rc '.dependencies' <<< "${combination}")

                COMBINATIONS+=("${PACKAGE}|test|${python}|${dependencies}")
            done < <(
                jq 'unique_by([.python, .dependencies]) | .[]' "${CWD}/../${PACKAGE}/supported-versions.json" -c
            )
        done

        unset combination
        unset python
        unset dependencies
        unset PACKAGES
        unset PACKAGE
        shift 2
        ;;
    "--sphinx-requirements")
        COMBINATIONS+=('skore|sphinx|3.14|["scikit-learn==1.8.*"]')
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
        IFS="|" read -r PACKAGE EXTRA PYTHON DEPENDENCIES <<< "${combination}"

        # Escape dependencies to be used in filename, both in Linux and Windows:
        # - make a str based on the JSON array
        # - replace `==` by `-`
        # - remove all `.*`
        ESCAPED=$(jq 'join("_and_") | gsub("=="; "-") | gsub("\\.\\*"; "")' -rc <<< "${DEPENDENCIES}")

        echo "Generating ${PACKAGE} ${EXTRA}-requirements: python==${PYTHON} | ${DEPENDENCIES} (${counter}/${#COMBINATIONS[@]})"

        # Copy everything necessary to compile requirements in `TMPDIR`
        mkdir -p "${TMPDIR}/${PACKAGE}"; cp "${CWD}/../${PACKAGE}/pyproject.toml" "${TMPDIR}/${PACKAGE}"

        # Force the dependencies by creating file overriding requirements
        > "${PACKAGE}/overrides.txt"

        for dependency in $(jq '.[]' -rc <<< "${DEPENDENCIES}"); do
            echo "${dependency}" >> "${PACKAGE}/overrides.txt"
        done

        FILEPATH="${CWD}/requirements/${PACKAGE}/python-${PYTHON}/${ESCAPED}/${EXTRA}-requirements.txt"

        # Create the requirements file tree
        mkdir -p $(dirname "${FILEPATH}")

        # Create the `.python-version` file used by `Dependabot`
        echo "${PYTHON}" > "${CWD}/requirements/${PACKAGE}/python-${PYTHON}/.python-version"

        # Create the requirements file
        uv python install --quiet "${PYTHON}"
        uv pip compile \
           --python-platform "linux" \
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
