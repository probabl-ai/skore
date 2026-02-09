.. list-table::
   :widths: 30 70
   {% for class_info in classes %}
   * - :class:`{{ class_info.name }}`
     - {{ class_info.doc }}

       .. dropdown:: Accessor methods
          :icon: chevron-down
          {% for accessor_name, accessor_data in class_info.accessors.items() %}
          **{{ accessor_name.capitalize() }}**: :class:`{{ class_info.name }}.{{ accessor_name }}`

          .. list-table::
             :widths: 30 70
             {% for method_name, method_doc in accessor_data.methods %}
             * - :func:`~{{ class_info.name }}.{{ accessor_name }}.{{ method_name }}`
               - {{ method_doc }}
			 {% endfor %}
		  {% endfor %}
   {% endfor %}
