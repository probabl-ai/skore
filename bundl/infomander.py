import json
from time import time
from diskcache import Cache
from joblib import dump
from pathlib import Path


class InfoMander:
    """Represents a dictionary, on disk, with a path-like structure."""
    def __init__(self, *path):
        self.project_path = "/".join(['.datamander'] + list(path))
        self.cache = Cache(self.project_path + '/stats')
        self.artifact_path = Path(self.project_path + '/artifacts')
        self.log_path = Path(self.project_path + '/logs')
        if '_artifacts' not in self.cache:
            self.cache['_artifacts'] = {}
        if '_logs' not in self.cache:
            self.cache['_logs'] = {}
        if '_templates' not in self.cache:
            self.cache['_templates'] = {}
        
    def add_info(self, key, value, method='overwrite'):
        if method == 'overwrite':
            self.cache[key] = value
        if method == 'append':
            if not isinstance(value, list):
                value = [value]
            if key in self.cache:
                value = value + self.cache[key]
            self.cache[key] = value
        self.cache['updated_at'] = int(time())

    def _add_to_key(self, top_key, key, value):
        orig = self.cache[top_key] 
        orig[key] = value
        self.add_info(top_key, orig)

    def add_artifact(self, key, obj, **metadata):
        file_location = self.artifact_path / f'{key}.joblib'
        if not file_location.parent.exists():
            file_location.parent.mkdir(parents=True)
        dump(obj, file_location)
        self._add_to_key('_artifacts', key, {'path': file_location, **metadata})

    def add_view(self, key, html):
        self._add_to_key('_templates', key, html)

    def add_logs(self, key, logs):
        self._add_to_key('_logs', key, logs)
                    
    def fetch(self):
        return {k: self.cache[k] for k in self.cache.iterkeys()}
    
    def __getitem__(self, key):
        return self.cache[key]
