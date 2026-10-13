.. _cli:

========================
Command-line interface
========================

.. currentmodule:: skore

The ``skore`` command synchronizes report projects, installs open-source agent
skills, and connects a coding agent to Skore Hub. Option-level detail lives in
the :ref:`command reference <cli-reference>`.

Hub commands need the ``hub`` extra (``pip install skore[hub]``). Synchronizing
with MLflow needs the ``mlflow`` extra (``pip install skore[mlflow]``).

.. _cli-sync:

Synchronizing projects
----------------------

``skore sync`` runs :meth:`Project.sync` from the shell. The positional argument
is the source project name. Pass ``--from`` or ``--to`` (or both) with one of
``local``, ``hub``, and ``mlflow``. The omitted side defaults to ``local``.

.. code-block:: bash

   $ skore sync my-xp --to hub --to-workspace my-workspace

   $ skore sync my-xp --from hub --from-workspace my-workspace --dry-run

   $ skore sync my-xp --to mlflow --tracking-uri http://localhost:5000 --both

``--from-workspace`` and ``--to-workspace`` are required for a Hub endpoint and
optional for a local one. Omit the local workspace to use the same discovery
rules as :class:`Project` (``SKORE_WORKSPACE``, a nearby ``skore`` directory, or
a new ``skore`` directory at the Git root). A workspace option is rejected for
MLflow. ``--tracking-uri`` sets the tracking URI of every MLflow endpoint in the
command; two MLflow projects share that URI. ``--to-project`` renames the
destination; otherwise it keeps the source name. ``--hub-url`` selects the Hub
(an API URL, or a frontend such as ``https://skore.probabl.ai``, which is
resolved through ``/.well-known/skore-hub.json``). It defaults to
``SKORE_HUB_URI`` or the public Hub.

``--both`` transfers missing reports in both directions. ``--dry-run`` prints
the plan and leaves both projects unchanged. Reports are matched by
``report_id``, as described in :ref:`synchronizing-projects`. The command
prints the dataframe returned by :meth:`Project.sync`.

.. _cli-skills:

Installing agent skills
-----------------------

``skore skills`` installs `Agent Skills <https://github.com/probabl-ai/skills>`_
from the latest GitHub release of ``probabl-ai/skills``. A skill id installs
that skill. A workflow id installs every skill the workflow includes. Pass
``--repo owner/name`` to install from another catalog; in the interactive
installer that value is pre-filled and can still be edited.

Run ``skore skills install`` with no arguments to open the installer. It asks
for the GitHub source, the skills, the target agent, and whether to install
into the project or the user directory. When the process has no terminal,
``CI`` is set, or a coding agent is detected, the same command prints the
catalog and the commands below.

.. code-block:: bash

   $ skore skills install all

   $ skore skills install <skill-or-workflow-id>

   $ skore skills install all --agent claude-code --agent cursor

   $ skore skills install all --global

   $ skore skills list

   $ skore skills update --all

   $ skore skills remove <skill-id>

With no ``--agent``, an explicit install uses the detected agent's project
directory, or ``.agents/skills`` when no agent is detected. ``--global``
installs into the user directory (for the default agent,
``~/.agents/skills``). ``--agent`` can be repeated. ``list``, ``update``, and
``remove`` scan every known agent unless ``--agent`` restricts them.
``update`` and ``remove`` with no ids open a picker of installed skills; pass
ids or ``--all`` when the process is non-interactive. ``remove --yes`` skips
the confirmation. Each installed skill records the repository it came from,
and ``update`` fetches that repository's latest release.

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - ``--agent``
     - Project directory
     - User directory (``--global``)
   * - ``agents`` (default)
     - ``.agents/skills``
     - ``~/.agents/skills``
   * - ``claude-code``
     - ``.claude/skills``
     - ``~/.claude/skills``
   * - ``cursor``
     - ``.cursor/skills``
     - ``~/.cursor/skills``
   * - ``codex``
     - ``.agents/skills``
     - ``~/.agents/skills``
   * - ``gemini``
     - ``.agents/skills``
     - ``~/.gemini/skills``
   * - ``windsurf``
     - ``.windsurf/skills``
     - ``~/.codeium/windsurf/skills``
   * - ``cline``
     - ``.cline/skills``
     - ``~/.cline/skills``
   * - ``roo``
     - ``.roo/skills``
     - ``~/.roo/skills``
   * - ``amp``
     - ``.agents/skills``
     - ``~/.config/agents/skills``
   * - ``github-copilot``
     - ``.github/skills``
     - project directory only
   * - ``bob``, ``bob-ide``
     - ``.bob/skills``
     - ``~/.bob/skills``

.. _cli-hub:

Using Skore Hub
---------------

Managing credentials
^^^^^^^^^^^^^^^^^^^^

``skore hub api-key`` stores a workspace-scoped API key in the local credential
registry. Generating a key opens a device login (a browser URL and a one-time
code) and waits up to ``--login-timeout`` seconds, 600 by default.

.. code-block:: bash

   $ skore hub api-key generate --workspace my-workspace

   $ skore hub api-key generate --workspace my-workspace --expires 3 --name ci

   $ skore hub api-key list

   $ skore hub api-key revoke --workspace my-workspace

``--expires`` is ``1``, ``3``, ``6`` (months), or ``never``. ``--name`` is the
name stored on the Hub; a random name is used when it is omitted. ``--force``
revokes the stored key for that host and workspace, then mints a new one.
``--host`` selects the Hub and accepts the same frontend URLs as
``skore sync --hub-url``. ``list`` prints the host and workspace of each stored
key. ``revoke`` removes the key locally and on the Hub.

:class:`Project` in ``hub`` mode still prefers ``SKORE_HUB_API_KEY`` over this
registry. See :ref:`project`.

Connecting a coding agent
^^^^^^^^^^^^^^^^^^^^^^^^^

``skore agent`` logs in to the Hub, records the workspace in ``.skore``, writes
the harness configuration, and launches the harness. The API key comes from the
registry above; the command mints one when that workspace has none.

On the first run, pick a workspace and a harness in the terminal. The command
saves ``hub_url``, ``workspace``, ``workspace_id``, and ``harness`` to
``.skore`` in the project directory (``--workspace``, defaulting to the current
directory) and adds ``.skore`` to ``.gitignore``. Later runs reuse that file.
``--hub-url`` points at another Hub and signs in again. ``--harness`` selects
the harness and updates the saved value. ``--model-id`` is the model id
advertised by the Hub (``skore-agent`` by default).

.. code-block:: bash

   $ skore agent

   $ skore agent --harness cursor

   $ skore agent --hub-url https://skore.probabl.ai --harness claude

When the command is already running inside the selected harness, it writes the
configuration and asks you to restart that harness or start a new session.
Otherwise it launches the harness.

When the process is non-interactive, omit ``--harness`` to use a detected
installed harness, or the harness saved in ``.skore``. Pass ``--harness``
when neither is available.

Supported ``--harness`` values:

- ``bob`` (Bob Shell) and ``bob-ide`` (Bob IDE; ``bobide`` is accepted too)
- ``claude`` (Claude CLI; ``claude-cli`` is accepted too) and ``claude-ui``
- ``claude-plugin``, which asks which IDE to open. ``claude-plugin-cursor``,
  ``claude-plugin-code``, and ``claude-plugin-code-insiders`` select that IDE
- ``cursor`` (Cursor IDE) and ``cursor-cli``
- ``codex`` (Codex CLI), ``opencode``, and ``pi``
- ``copilot`` (Copilot in VS Code) and ``copilot-cli`` (``github-copilot-cli``
  is accepted too)

Detection uses ``PATH``, the application bundle on macOS for Bob IDE and Claude
UI, or the Claude Code IDE extension for the Claude plugin.
