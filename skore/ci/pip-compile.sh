#!/bin/bash

CWD="$PWD"
TMPDIR=$(mktemp -d); trap 'rm -rf ${TMPDIR}' 0

declare -a COMBINATIONS

COMBINATIONS[0]='3.9;1.6'
COMBINATIONS[1]='3.10;1.6'
COMBINATIONS[2]='3.11;1.6'
COMBINATIONS[3]='3.12;1.4'
COMBINATIONS[4]='3.12;1.5'
COMBINATIONS[5]='3.12;1.6'

set -eu

(
    cp -r .. "${TMPDIR}/skore"
    cp ../../LICENSE "${TMPDIR}/LICENSE"
    cp ../../README.md "${TMPDIR}/README.md"

    cd "${TMPDIR}"

    for combination in "${COMBINATIONS[@]}"
    do
        IFS=";" read -r -a combination <<< "${combination}"

        python="${combination[0]}"
        scikit_learn="${combination[1]}"
        filepath="${CWD}/requirements/python-${python}-scikit-learn-${scikit_learn}-test-requirements.txt"

        sed -i "s/scikit-learn.*/scikit-learn==${scikit_learn}.*/g" skore/test-requirements.in

        pyenv local "${python}"
        python -m venv "python-${python}"
        source "python-${python}/bin/activate"

        echo "Generating requirements: python ==${python} | scikit-learn ==${scikit_learn}"

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
