{{ objname | escape | underline(line="=") }}

.. currentmodule:: {{ module.split('.')[0] }}

.. autoaccessor:: {{ (module.split('.')[1:] + [objname]) | join('.') }}

{% set accessor_path = (module.split('.')[1:] + [objname]) | join('.') -%}
{% if accessor_toctrees is defined and accessor_path in accessor_toctrees %}
.. toctree::
   :hidden:
{% for entry in accessor_toctrees[accessor_path] %}
   {{ entry }}
{%- endfor %}

{% endif %}
