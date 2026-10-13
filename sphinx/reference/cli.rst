.. _cli-reference:

Command-line interface
----------------------

The ``skore`` command synchronizes report projects, installs agent skills, and
connects a coding agent to Skore Hub. Usage is described in the
:ref:`user guide <cli>`.

.. click:: skore._cli:cli
   :prog: skore
   :nested: short

.. click:: skore._cli.sync:sync
   :prog: skore sync

.. click:: skore._cli.skills:skills
   :prog: skore skills
   :nested: full

.. click:: skore._cli.hub:hub
   :prog: skore hub
   :nested: full

.. click:: skore._cli.agent:agent
   :prog: skore agent
