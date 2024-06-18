# mandr

Service to send data into, install via

```
python -m pip install -e
python -m pip install flask diskcache pandas altair
```


```python
from mandr import InfoMander

# We use paths to explain where the dictionary is stored
mander = InfoMander('/org/usecase/train/1')
mander.add_info(...)
mander.add_logs(...)
mander.add_templates(...)
mander.add_views(...)
```

When ready to view, run flask:

```
python -m flask --app mandr.app run --reload
```
