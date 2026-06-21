"""
.. _example_skore_sync_project:

====================================
Synchronize projects across backends
====================================

This example shows how to use :meth:`~skore.Project.sync_with` to reconcile
reports between projects. The typical workflow is to persist reports locally
while offline, then push them to Skore Hub once you are back online.

The first part uses two local workspaces so the example runs without network
access. The same API applies when the destination is a hub or MLflow project.
"""

# %%
# Work offline with a local project
# =================================
from pathlib import Path
from tempfile import TemporaryDirectory

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from skore import Project, evaluate

X, y = make_regression(random_state=42)
report = evaluate(LinearRegression(), X, y, splitter=0.2)

offline_dir = TemporaryDirectory()
offline = Project("offline-xp", workspace=Path(offline_dir.name))
offline.put("baseline", report)

# %%
# Put to another backend when connectivity is available
# =====================================================
#
# Here we simulate uploading to a remote backend with a second local workspace.
# Replace ``remote`` with ``Project("workspace/project", mode="hub")`` after
# calling ``skore.login(mode="hub")``, or with an MLflow experiment project.
remote_dir = TemporaryDirectory()
remote = Project("remote-xp", workspace=Path(remote_dir.name))

result = offline.sync_with(remote, direction="put")
result.summary()

# %%
# Inspect the remote project summary
remote.summarize()

# %%
# Bidirectional reconcile
# =======================
#
# ``direction="both"`` merges changes from both sides. Re-running sync skips
# reports that are already synchronized.
offline.sync_with(remote, direction="both").summary()

offline_dir.cleanup()
remote_dir.cleanup()
