#!/usr/bin/env bash
#
# This script compiles all the `test-requirements.txt` files, based on combinations of
# `python` and `scikit-learn` versions. These combinations mirror those defined in the
# GitHub workflows.
#
# You can pass any `uv pip compile` parameter:
#
#     $ bash pip-compile.sh <skore|skore-remote-project> --upgrade
#

CWD=$(realpath $(dirname $0))
TMPDIR=$(mktemp -d)
PACKAGE=$1

case "${PACKAGE}" in
    "skore"|"skore-remote-project") ;;
    *)
        >&2 echo "Error: Unknown package: '${PACKAGE}'"
        >&2 echo "Usage: bash pip-compile.sh <skore|skore-remote-project> [option...]"
        exit 1
        ;;
esac

# Make sure that `TMPDIR` is removed on exit, whatever the signal
trap 'rm -rf ${TMPDIR}' 0

# Declare the combinations of `python` and `scikit-learn` versions
declare -a COMBINATIONS

COMBINATIONS[0]='3.9;1.6'
COMBINATIONS[1]='3.10;1.6'
COMBINATIONS[2]='3.11;1.6'
COMBINATIONS[3]='3.12;1.4'
COMBINATIONS[4]='3.12;1.5'
COMBINATIONS[5]='3.12;1.6'

set -eu

(
    # Copy everything necessary to compile requirements in `TMPDIR`
    mkdir "${TMPDIR}/${PACKAGE}"; cp "${CWD}/../${PACKAGE}/pyproject.toml" "${TMPDIR}/${PACKAGE}"

    # Move to `TMPDIR` to avoid absolute paths in requirements file
    cd "${TMPDIR}"

    counter=1
    for combination in "${COMBINATIONS[@]}"
    do
        IFS=";" read -r -a combination <<< "${combination}"

        python="${combination[0]}"
        scikit_learn="${combination[1]}"
        filepath="${CWD}/requirements/${PACKAGE}/python-${python}/scikit-learn-${scikit_learn}/test-requirements.txt"

        echo "Generating ${PACKAGE} requirements: python==${python} | scikit-learn==${scikit_learn} (${counter}/${#COMBINATIONS[@]})"

        # Force the `scikit-learn` version by creating file overriding requirements
        echo "scikit-learn==${scikit_learn}.*" > "${PACKAGE}/overrides.txt"

        # Create the requirements file tree
        mkdir -p $(dirname "${filepath}")

        # Create the `.python-version` file used by `Dependabot`
        echo "${python}" > "${CWD}/requirements/${PACKAGE}/python-${python}/.python-version"

        # Create the requirements file
        uv pip compile \
           --quiet \
           --no-strip-extras \
           --no-header \
           --extra=test \
           --override "${PACKAGE}/overrides.txt" \
           --python-version "${python}" \
           --output-file "${filepath}" \
           "${PACKAGE}/pyproject.toml" \
           "${@:2}"

        let counter++
    done
)
