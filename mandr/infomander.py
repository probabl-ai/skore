import json
from time import time
from diskcache import Cache
from joblib import dump
from pathlib import Path
from rich.console import Console
from .templates import TemplateRenderer

console = Console()
LOGS_KEY = 'logs'
VIEWS_KEY = 'views'
TEMPLATES_KEY = 'templates'
ARTIFACTS_KEY = 'artifacts'

STATS_FOLDER = '.stats'
ARTIFACTS_FOLDER = '.artifacts'
LOGS_FOLDER = '.logs'

class InfoMander:
    """Represents a dictionary, on disk, with a path-like structure."""
    def __init__(self, path):        
        # Set local disk paths
        self.project_path = Path('.datamander/' + path)
        self.cache = Cache(self.project_path / STATS_FOLDER)

        # For practical reasons the logs and artifacts are stored on disk, not sqlite
        # We could certainly revisit this later though
        self.artifact_path = self.project_path / ARTIFACTS_FOLDER
        self.log_path = self.project_path / LOGS_FOLDER

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
        self._add_to_key(VIEWS_KEY, key, html)

    def add_template(self, key, template):
        if key in self.cache[VIEWS_KEY].keys():
            raise ValueError(f'Cannot add template {key} because there is already a view with the same name.')
        self._add_to_key('_templates', key, template)

    def render_templates(self):
        for name, template in self.cache['_templates'].items():
            self.add_view(name, template.render(self))

    def add_logs(self, key, logs):
        self._add_to_key(LOGS_KEY, key, logs)
                    
    def fetch(self):
        return {k: self.cache[k] for k in self.cache.iterkeys()}
    
    def __getitem__(self, key):
        return self.cache[key]

    @classmethod
    def get_property(cls, mander, dsl_str):
        # Note that we may be dealing with a nested retreival
        prop_chain = [e for e in dsl_str.split('.') if e]
        if len(prop_chain) == 1:
            without_dot = prop_chain[0]
            return mander.cache[without_dot]
        
        # Handle special case with getting a property from direct children
        if prop_chain[0].startswith('*'):
            return [child.get('@mander' + '.'.join(prop_chain[1:])) for child in mander.children()]
        
        # At this point we know for sure it's just a nested property in a single mander
        item_of_interest = mander.cache[prop_chain[0]]
        for prop in prop_chain[1:]:
            item_of_interest = item_of_interest[prop]
        return item_of_interest

    def dsl_path_exists(self, path):
        actual_path = '.datamander' / path  
        assert Path(actual_path).exists()
    
    def get_child(self, *path):
        new_path = self.project_path
        for p in path:
            new_path = new_path / p
        return InfoMander(str(new_path))

    def children(self):
        return [InfoMander('/'.join(p.parts[1:])) for p in self.project_path.iterdir() if p.is_dir() and not p.name.startswith('.')]

    def get(self, dsl_str):
        if '@mander' not in dsl_str:
            raise ValueError('@mander is needed at the start of dsl string')
        path = [p for p in dsl_str.replace('@mander', '').split('/') if p]
        # There is no path to another mander, but we may have a nested property
        if len(path) == 1:
            if path[0] == '*':
                return self.children()
            return InfoMander.get_property(self, path[0])
        mander = self.get_child(*path[:-1])
        return mander.get_property(mander, path[-1])
    
    def __repr__(self):
        return f'InfoMander({self.project_path})'
