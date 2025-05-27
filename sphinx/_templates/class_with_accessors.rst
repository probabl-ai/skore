{{ objname | escape | underline(line="=") }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :exclude-members: metrics, feature_importance, reports
   :inherited-members:
   :special-members: __call__

.. minigallery:: {{ module }}.{{ objname }} {% for meth in methods %}{{ module }}.{{ objname }}.{{ meth }} {% endfor %}
   :add-heading: Gallery examples
   :heading-level: -
