#!/bin/bash

set -eu

BRANCH="documentation-preview"
ARTIFACTS=$(
    gh api \
       --paginate \
       -H "Accept: application/vnd.github+json" \
       -H "X-GitHub-Api-Version: 2022-11-28" \
       /repos/probabl-ai/skore/actions/artifacts \
    |
    jq -c ".artifacts[] | select(.workflow_run.head_branch == \"${BRANCH}\") | {id: .id, name: .name}"
)

for ARTIFACT in $ARTIFACTS; do
    ID=$(echo "${ARTIFACT}" | jq -r '.id')
    NAME=$(echo "${ARTIFACT}" | jq -r '.name')

    echo "Deleting artifact (BRANCH: \"${BRANCH}\", NAME: \"${NAME}\", ID: \"${ID}\")"

    gh api \
       --method DELETE \
       --silent \
       -H "Accept: application/vnd.github+json" \
       -H "X-GitHub-Api-Version: 2022-11-28" \
       /repos/probabl-ai/skore/actions/artifacts/${ID}
done
