.. _install:

Install
=======

.. currentmodule:: skore

.. raw:: html

    <div class="container mt-4">

    <ul class="nav nav-pills nav-fill" id="installation" role="tablist">
        <li class="nav-item" role="presentation">
            <a class="nav-link active" id="pip-tab" data-bs-toggle="tab" data-bs-target="#pip-tab-pane" type="button" role="tab" aria-controls="pip" aria-selected="true">Using pip</a>
        </li>
        <li class="nav-item" role="presentation">
            <a class="nav-link" id="conda-tab" data-bs-toggle="tab" data-bs-target="#conda-tab-pane" type="button" role="tab" aria-controls="conda" aria-selected="false">Using conda</a>
        </li>
    </ul>

    <div class="tab-content">
        <div class="tab-pane fade show active" id="pip-tab-pane" role="tabpanel" aria-labelledby="pip-tab" tabindex="0">
            <hr />

We recommend using a `virtual environment (venv) <https://docs.python.org/3/tutorial/venv.html>`_.
You need ``python>=3.9``.

Then, run:

.. code-block:: bash

    pip install -U skore

You can check skore's latest version on `PyPI <https://pypi.org/project/skore/>`_.

.. raw:: html

    </div>
    <div class="tab-pane fade" id="conda-tab-pane" role="tabpanel" aria-labelledby="conda-tab" tabindex="0">
        <hr />

Skore is available in ``conda-forge``:

.. code-block:: bash

    conda install conda-forge::skore

You can find information on the latest version `here <https://anaconda.org/conda-forge/skore>`_.

.. raw:: html

        </div>
    </div>
    </div>