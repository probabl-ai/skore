.. _install:

Install
=======

.. currentmodule:: skore

.. tab-set::

    .. tab-item:: Using pip

        We recommend using a `virtual environment (venv) <https://docs.python.org/3/tutorial/venv.html>`_.
        You need ``python>=3.9``.

        Then, if you just want to use Skore Lib locally, run:

        .. code-block:: bash

            pip install -U skore

        *Alternatively*, if you wish to use Skore locally and also interact with Skore
        Hub, run:

        .. code-block:: bash

            pip install -U skore[hub]

        You can check Skore Lib's latest version on `PyPI <https://pypi.org/project/skore/>`_.

    .. tab-item:: Using conda

        Skore is available in ``conda-forge``:

        .. code-block:: bash

            conda install conda-forge::skore

        You can find information on the latest version `here <https://anaconda.org/conda-forge/skore>`_.
