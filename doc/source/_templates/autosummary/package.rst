{{ fullname | escape | underline }}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ fullname  }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
