{{ objname | escape | underline(line="=") }}

.. currentmodule:: {{ module.split('.')[0] }}

.. autoaccessor:: {{ (module.split('.')[1:] + [objname]) | join('.') }}
