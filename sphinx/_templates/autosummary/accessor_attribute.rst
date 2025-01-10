{{ objname | escape | underline(line="=") }}

.. currentmodule:: {{ module.split('.')[0] }}

.. autoaccessorattribute:: {{ (module.split('.')[1:] + [objname]) | join('.') }}
