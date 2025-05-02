---
date: 2025-04-24
decision-makers: ["@thomass-dev"]
---

# Use merge queue in CI

## Context and Problem Statement

With external contributions related to ESOC, the number of pull-request is growing fast.
The `main` branch has a particular restriction `Require branches to be up to date before
merging`. It ensures that pull requests have been tested with the latest code.

Even if it is a good practice, it begins to have an impact on maintainers velocity. It
implies updating each pull request after any modification to the `main` branch.

For example, suppose that there are several PRs that are done and ready to merge. With
the current system, it is not possible to simply go through the PRs and merge them; one
needs to, for each PR:
- Put a lock on the PR merges (`main` should not move until the PR is merged),
- Update the PR with `main`,
- Wait for CI to pass (~10 min today),
- Merge,
- Release the lock.

## Decision Outcome

We decide to add a merge queue to the `main` branch.

https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue

> A merge queue helps increase velocity by automating pull request merges into a busy
> branch and ensuring the branch is never broken by incompatible changes.
>
> The merge queue provides the same benefits as the *Require branches to be up to date*
> before merging branch protection, but does not require a pull request author to update
> their pull request branch and wait for status checks to finish before trying to merge.
>
> Using a merge queue is particularly useful on branches that have a relatively high
> number of pull requests merging each day from many different users.
>
> Once a pull request has passed all required branch protection checks, a user with
> write access to the repository can add the pull request to the queue. The merge queue
> will ensure the pull request's changes pass all required status checks when applied to
> the latest version of the target branch and any pull requests already in the queue.

### Consequences

We have to make some changes to prepare the setup of the GH merge queue, for which we
need to specify the jobs required to succeed (as opposed to specifying global workflow).

For a job to be required, its workflow must be executed and not skipped. Currently, we
skip entire workflow via path filtering to avoid unnecessary calculations, which is
incompatible with "required jobs". Thus, we have to change this behaviour to always
execute workflows, but skip jobs individually based on modified files.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/troubleshooting-required-status-checks#handling-skipped-but-required-checks

To avoid specifying every required job individually, we have to implement "all-green"
jobs that centralize the results of entire workflows, success or fail. Only these jobs
will be marked as required. For example, the only required job of the `backend` workflow
would be a combination of the results of linting and testing.

Additionally, the CI setup we had before managing forks, with a master "ci-all-green"
workflow, will not work with merge queues: the chain `workflow -> workflow_call ->
workflow_run` is not triggered.

## More Information

Implementation in pull-request [#1514](https://github.com/probabl-ai/skore/pull/1514).
