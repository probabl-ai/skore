#!/bin/bash
#
# This script compiles all the `test-requirements.txt` files, based on combinations of
# `python` and `scikit-learn` versions. These combinations mirror those defined in the
# GitHub `backend` workflow.
#
# You can pass any piptools parameter:
#
#     $ bash pip-compile.sh --upgrade
#

CWD="$PWD"
TMPDIR=$(mktemp -d)

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
    cp -r .. "${TMPDIR}/skore"
    cp ../../LICENSE "${TMPDIR}/LICENSE"
    cp ../../README.md "${TMPDIR}/README.md"

    # Move to `TMPDIR` to avoid absolute paths in requirements file
    cd "${TMPDIR}"

    for combination in "${COMBINATIONS[@]}"
    do
        IFS=";" read -r -a combination <<< "${combination}"

        python="${combination[0]}"
        scikit_learn="${combination[1]}"
        filepath="${CWD}/requirements/python-${python}/scikit-learn-${scikit_learn}/test-requirements.txt"

        echo "Generating requirements: python ==${python} | scikit-learn ==${scikit_learn}"

        # Force the `python` version by creating the appropriate virtual environment
        pyenv local "${python}"
        python -m venv "python-${python}"
        source "python-${python}/bin/activate"

        # Force the `scikit-learn` version by overloading test requirements
        sed -i "s/scikit-learn.*/scikit-learn==${scikit_learn}.*/g" skore/test-requirements.in

        # Create the requirements file tree
        mkdir -p $(dirname "${filepath}")

        # Create the requirements file
        python -m pip install --upgrade pip pip-tools --quiet
        python -m piptools compile \
               --quiet \
               --no-strip-extras \
               --no-header \
               --extra=test \
               --output-file="${filepath}" \
               "${@:2}" \
               skore/pyproject.toml
    done
)
