# mandr

Service to send data into, install via

```
python -m pip install -e
python -m pip install flask diskcache pandas altair
```


```python
from bundl.infomander import InfoMander

mander = InfoMander()
mander.add_info(...)
mander.add_logs(...)
mander.add_templates(...)
```

When ready to view, run flask:

```
python -m flask --app bundl.app run --reload
```
