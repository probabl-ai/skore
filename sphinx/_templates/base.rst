{{ objname | escape | underline(line="=") }}

{% if objtype == "module" -%}

.. automodule:: {{ fullname }}

{%- elif objtype == "function" -%}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}
   :noindex:

.. minigallery:: {{ module }}.{{ objname }}
   :add-heading: Gallery examples
   :heading-level: -

{%- elif objtype == "class" -%}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :special-members: __call__
   :noindex:

.. minigallery:: {{ module }}.{{ objname }} {% for meth in methods %}{{ module }}.{{ objname }}.{{ meth }} {% endfor %}
   :add-heading: Gallery examples
   :heading-level: -

{%- else -%}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}

{%- endif -%}
