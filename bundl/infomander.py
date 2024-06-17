import json
from time import time
from diskcache import Cache
from joblib import dump
from pathlib import Path
from .templates import TemplateRenderer


LOGS_KEY = '_logs'
VIEWS_KEY = '_views'
TEMPLATES_KEY = '_templates'
ARTIFACTS_KEY = '_artifacts'
STATS_KEY = '_stats'

class InfoMander:
    """Represents a dictionary, on disk, with a path-like structure."""
    def __init__(self, path):
        # This is a bit of a stub. I assume a local:// prefix to the path when we
        # are dealing with a local path and we can also replace it with a cloud path later.
        # This isn't implemented at all though.
        if not path.startswith("local://"):
            raise ValueError("Only local:// paths are supported for now.")
        
        # Set local disk paths
        self.project_path = Path(path.replace('local://', '.datamander'))
        self.cache = Cache(self.project_path / STATS_KEY)

        # For practical reasons the logs and artifacts are stored on disk, not sqlite
        # We could certainly revisit this later though
        self.artifact_path = self.project_path / ARTIFACTS_KEY
        self.log_path = self.project_path / LOGS_KEY

        # Initialize the internal cache with empty values if need be
        for key in [ARTIFACTS_KEY, TEMPLATES_KEY, VIEWS_KEY, LOGS_KEY]:
            if key not in self.cache:
                self.cache[key] = {}
        
        # This will be used for rendering templates into views
        self.renderer = TemplateRenderer(self)
        
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
        self._add_to_key(ARTIFACTS_KEY, key, {'path': file_location, **metadata})

    def add_view(self, key, html):
        self._add_to_key('_views', key, html)

    def add_template(self, key, template):
        if key in self.cache['_view'].keys():
            raise ValueError(f'Cannot add template {key} because there is already a view with the same name.')
        self._add_to_key('_templates', key, template)

    def render_templates(self):
        for name, template in self.cache['_templates'].items():
            self.add_view(name, template.render(self))

    def add_logs(self, key, logs):
        self._add_to_key('_logs', key, logs)
                    
    def fetch(self):
        return {k: self.cache[k] for k in self.cache.iterkeys()}
    
    def __getitem__(self, key):
        return self.cache[key]
