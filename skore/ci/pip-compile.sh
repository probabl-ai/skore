#!/bin/bash
#
# This script compiles all the `test-requirements.txt` files, based on combinations of
# `python` and `scikit-learn` versions. These combinations mirror those defined in the
# GitHub `backend` workflow.
#
# You can pass any `uv pip compile` parameter:
#
#     $ bash pip-compile.sh --upgrade
#

CWD=$(realpath $(dirname $0))
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
    mkdir "${TMPDIR}/skore"; cp "${CWD}/../pyproject.toml" "${TMPDIR}/skore"

    # Move to `TMPDIR` to avoid absolute paths in requirements file
    cd "${TMPDIR}"

    counter=1
    for combination in "${COMBINATIONS[@]}"
    do
        IFS=";" read -r -a combination <<< "${combination}"

        python="${combination[0]}"
        scikit_learn="${combination[1]}"
        filepath="${CWD}/requirements/python-${python}/scikit-learn-${scikit_learn}/test-requirements.txt"

        echo "Generating requirements: python==${python} | scikit-learn==${scikit_learn} (${counter}/${#COMBINATIONS[@]})"

        # Force the `scikit-learn` version by creating file overriding requirements
        echo "scikit-learn==${scikit_learn}.*" > skore/overrides.txt

        # Create the requirements file tree
        mkdir -p $(dirname "${filepath}")

        # Create the requirements file
        uv pip compile \
           --quiet \
           --no-strip-extras \
           --no-header \
           --extra=test \
           --override skore/overrides.txt \
           --python "${python}" \
           --output-file "${filepath}" \
           skore/pyproject.toml \
           "${@:2}"

        let counter++
    done
)
