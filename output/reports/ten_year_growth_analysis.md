# Report Generation Failed

Error: 'growth_areas' is undefined

Traceback:
Traceback (most recent call last):
  File "/Users/vanessa/Library/CloudStorage/Dropbox/allCode/achipos/src/reporting/ten_year_growth_report.py", line 2133, in generate_report
    rendered_report = template.render(self.report_data)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/vanessa/Library/CloudStorage/Dropbox/allCode/achipos/.venv/lib/python3.11/site-packages/jinja2/environment.py", line 1295, in render
    self.environment.handle_exception()
  File "/Users/vanessa/Library/CloudStorage/Dropbox/allCode/achipos/.venv/lib/python3.11/site-packages/jinja2/environment.py", line 942, in handle_exception
    raise rewrite_traceback_stack(source=source)
  File "/Users/vanessa/Library/CloudStorage/Dropbox/allCode/achipos/src/templates/reports/ten_year_growth_analysis.md", line 96, in top-level template code
    {% if growth_areas.get('primary') %}
    ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/vanessa/Library/CloudStorage/Dropbox/allCode/achipos/.venv/lib/python3.11/site-packages/jinja2/environment.py", line 490, in getattr
    return getattr(obj, attribute)
           ^^^^^^^^^^^^^^^^^^^^^^^
jinja2.exceptions.UndefinedError: 'growth_areas' is undefined
