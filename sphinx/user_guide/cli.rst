.. _cli:

========================
Command-line interface
========================

.. currentmodule:: skore

The ``skore`` command synchronizes report projects, installs open-source agent
skills, and connects a coding agent to Skore Hub. Every option is listed in
the :ref:`command reference <cli-reference>`.

Hub commands need the ``hub`` extra (``pip install skore[hub]``). Synchronizing
with MLflow needs the ``mlflow`` extra (``pip install skore[mlflow]``).

.. _cli-sync:

Synchronizing projects
----------------------

``skore sync`` copies reports from one project to another. Start from a local
project named ``my-xp`` and send it to a Hub workspace:

.. code-block:: bash

   $ skore sync my-xp --to hub --to-workspace my-workspace

The name after ``skore sync`` is the source project. ``--to hub`` makes the
Hub the destination, and ``--to-workspace`` names that workspace. The source
stays local, so its workspace is the one :class:`Project` would discover.

To see what would come back from the Hub before anything is copied, reverse
the direction and stop after the plan:

.. code-block:: bash

   $ skore sync my-xp --from hub --from-workspace my-workspace --dry-run

Here the Hub project is the source and the local project is the destination.
``--dry-run`` prints the plan and leaves both projects unchanged.

The same command can target MLflow. This copies missing reports in both
directions between the local project and an MLflow tracking server:

.. code-block:: bash

   $ skore sync my-xp --to mlflow --tracking-uri http://localhost:5000 --both

``--tracking-uri`` is the MLflow server. ``--both`` fills in whichever side is
missing a report. How reports are matched is described in
:ref:`synchronizing-projects`.

At least one of ``--from`` or ``--to`` is required; an omitted endpoint defaults to
local storage. Use ``--to-project`` when the destination project has a different name.
Hub endpoints also accept ``--hub-url``, while ``--from-workspace`` and
``--to-workspace`` select endpoint-specific workspaces. See :ref:`cli-reference` for
the complete option list.

.. _cli-skills:

Installing agent skills
-----------------------

``skore skills`` installs `Agent Skills <https://github.com/probabl-ai/skore-skills>`_
from the latest release of ``probabl-ai/skore-skills``. To install every skill into
the current project:

.. code-block:: bash

   $ skore skills install --all

That writes them to ``.agents/skills``, or to the project directory of the
agent that is currently running. To install one skill or one workflow:

.. code-block:: bash

   $ skore skills install <skill-or-workflow-id>

A workflow id installs every skill that workflow includes. Running
``skore skills install`` with no arguments opens a picker for the source, the
skills, the agent, and the project or user directory. Without a terminal, that
same command prints the catalog instead.

To see what is installed:

.. code-block:: bash

   $ skore skills list

To bring those skills up to the latest release:

.. code-block:: bash

   $ skore skills update --all

To delete one skill:

.. code-block:: bash

   $ skore skills remove <skill-id>

The ``skore-skills`` package also provides deterministic helpers used by the
installed skills. They are available through ``skore skills tools`` as an alias
for the standalone ``skore-skills`` command:

.. code-block:: bash

   $ skore skills tools --help

.. _cli-hub:

Using Skore Hub
---------------

Managing credentials
^^^^^^^^^^^^^^^^^^^^

A Hub project and a Hub agent both use a workspace API key stored on your
machine. Create one for a workspace on a Hub host:

.. code-block:: bash

   $ skore hub api-key generate --host https://api.skore.probabl.ai --workspace my-workspace

``--host`` is the Hub backend address, not the frontend address. This opens a
device login in the browser, then stores the key for ``my-workspace``. Key
generation also accepts options for the key name, expiration, login timeout,
and replacing an existing key; see :ref:`cli-reference`.
See which keys are already stored:

.. code-block:: bash

   $ skore hub api-key list

And drop the key for a workspace, locally and on the Hub:

.. code-block:: bash

   $ skore hub api-key revoke --host https://api.skore.probabl.ai --workspace my-workspace

:class:`Project` in ``hub`` mode still prefers ``SKORE_HUB_API_KEY`` over this
registry. See :ref:`project`.

Connecting a coding agent
^^^^^^^^^^^^^^^^^^^^^^^^^

``skore agent`` signs in, chooses a workspace and a harness, and launches that
harness with the Hub configured. From the project directory:

.. code-block:: bash

   $ skore agent

The first run asks you to pick the workspace and the harness, saves that
choice in ``.skore``, and adds ``.skore`` to ``.gitignore``. It also creates
an API key when that workspace does not have one yet. Later runs reuse
``.skore`` and launch the harness.

If you already know which harness to open, name it:

.. code-block:: bash

   $ skore agent --harness cursor

Supported harnesses include Bob, Claude, Cursor, Codex, OpenCode, Pi, and
Copilot, with CLI and IDE variants where available. The accepted ``--harness`` values
are listed in :ref:`cli-reference`. When this command is already running inside the
selected harness, it writes the configuration and asks you to start a new session.
