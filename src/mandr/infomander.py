"""Contains the code for the main InfoMander class."""

from datetime import UTC, datetime
from pathlib import Path

from diskcache import Cache
from joblib import dump

from .templates import TemplateRenderer


def _get_storage_path() -> Path:
    """Return a path to the local mander storage."""
    return Path(".datamander")


class InfoMander:
    """Represents a dictionary, on disk, with a path-like structure."""

    ARTIFACTS_KEY = "artifacts"
    LOGS_KEY = "logs"
    TEMPLATES_KEY = "templates"
    VIEWS_KEY = "views"

    RESERVED_KEYS = {
        ARTIFACTS_KEY,
        LOGS_KEY,
        TEMPLATES_KEY,
        VIEWS_KEY,
    }

    STATS_FOLDER = ".stats"
    ARTIFACTS_FOLDER = ".artifacts"

    RESERVED_FOLDERS = {
        STATS_FOLDER,
        ARTIFACTS_FOLDER,
    }

    def __init__(self, path: str, /, *, root: Path = None):
        if Path(path).is_absolute():
            raise ValueError("Cant use absolute path")

        if root is None:
            root = Path(".datamander").resolve()

        # Set local disk paths
        self.path = path
        self.root = root
        self.project_path = self.root / path
        self.cache = Cache(self.project_path / InfoMander.STATS_FOLDER)

        # For practical reasons the artifacts are stored on disk, not sqlite
        # We could certainly revisit this later though
        self.artifact_path = self.project_path / InfoMander.ARTIFACTS_FOLDER

        # Initialize the internal cache with empty values if need be
        for key in InfoMander.RESERVED_KEYS:
            if key not in self.cache:
                self.cache[key] = {}

        # This will be used for rendering templates into views
        self.renderer = TemplateRenderer(self)

    @staticmethod
    def _check_key(key):
        if key in InfoMander.RESERVED_KEYS:
            raise ValueError(
                f"Cannot overwrite '{key}' key. This is reserved for internal use."
            )

    def add_info(self, key, value, overwrite=True):
        """
        Add information to the cache. Can be appended or overwritten.

        Parameters
        ----------
        key : str
            The key to add the information to.
        value : any
            The value to add.
        overwrite : bool, default=True
            Overwrite value, otherwise append.
        """
        if overwrite:
            self.cache[key] = value
        else:
            if not isinstance(value, list):
                value = [value]
            if key in self.cache:
                value = value + self.cache[key]
            self.cache[key] = value

        self.cache["updated_at"] = datetime.now(tz=UTC).isoformat()

    def _add_to_key(self, top_key, key, value):
        """
        Add a key to a dictionary that is stored in the cache.

        Sometimes we store a dictionary in the cache, this method allows us to add a new
        key to it.

        Parameters
        ----------
        top_key : str
            The key that points to the dictionary in the cache.
        key : str
            The key to add to the dictionary.
        value : any
            The value to add to the dictionary.
        """
        orig = self.cache[top_key]
        orig[key] = value
        self.add_info(top_key, orig)

    def add_artifact(self, key, obj, **metadata):
        """
        Add an artifact to the cache.

        Artifacts are treated differently as normal information objects. The main
        assumption here is that these may contain machine learning models and it can
        be a whole lot more practical to store these on disk.

        Parameters
        ----------
        key : str
            The key to store the artifact under.
        obj : any
            The object to store.
        metadata : dict
            Any metadata to store with the object.
        """
        self._check_key(key)
        file_location = self.artifact_path / f"{key}.joblib"
        if not file_location.parent.exists():
            file_location.parent.mkdir(parents=True)
        dump(obj, file_location)
        self._add_to_key(InfoMander.ARTIFACTS_KEY, key, metadata)

    def add_view(self, key, html):
        """
        Add a rendered view to the cache.

        Views can be thought of as static dashboards that we render based on the data
        that is stored in the mander. This view will become available in the web ui.

        Parameters
        ----------
        key : str
            The key to store the view under.
        html : str
            The HTML to store.
        """
        self._check_key(key)
        self._add_to_key(InfoMander.VIEWS_KEY, key, html)

    def add_template(self, key, template):
        """
        Add a template to the mander.

        Templates are used to render views. They are stored in the cache and can be
        rendered at a later stage.

        Parameters
        ----------
        key : str
            The key to store the template under.
        template : Template
            The template to store.
        """
        self._check_key(key)

        if key in self.cache[InfoMander.VIEWS_KEY]:
            raise ValueError(
                f"Unable to add template '{key}': view rendered with this key name."
            )

        self._add_to_key(InfoMander.TEMPLATES_KEY, key, template)

    def render_templates(self):
        """
        Render all templates in the cache, which becomes available as views.

        Parameters
        ----------
        key : str
            The key to store the template under.
        template : Template
            The template to store.
        """
        for name, template in self.cache[InfoMander.TEMPLATES_KEY].items():
            self.add_view(name, template.render(self))

    def add_logs(self, key, logs):
        """
        Add logs to the cache.

        Logs are stored in the cache and can be viewed in the web ui.

        Parameters
        ----------
        key : str
            The key to store the logs under.
        logs : list
            The logs to store.
        """
        self._add_to_key(InfoMander.LOGS_KEY, key, logs)

    def fetch(self):
        """Return all information from the cache."""
        return {k: self.cache[k] for k in self.cache.iterkeys()}

    def __getitem__(self, key):
        """Return a specific item from the cache."""
        return self.cache[key]

    @staticmethod
    def get_property(mander, dsl_str):
        """
        Get a property from a mander using the DSL syntax.

        Parameters
        ----------
        mander : InfoMander
            The mander to get the property from.
        dsl_str : str
            The DSL string to use.
        """
        # Note that we may be dealing with a nested retrieval
        prop_chain = [e for e in dsl_str.split(".") if e]
        if len(prop_chain) == 1:
            without_dot = prop_chain[0]
            return mander.cache[without_dot]

        # Handle special case with getting a property from direct children
        if prop_chain[0].startswith("*"):
            return [
                child.get("@mander" + ".".join(prop_chain[1:]))
                for child in mander.children()
            ]

        # At this point we know for sure it's just a nested property in a single mander
        item_of_interest = mander.cache[prop_chain[0]]
        for prop in prop_chain[1:]:
            item_of_interest = item_of_interest[prop]
        return item_of_interest

    def dsl_path_exists(self, path):
        """
        Assert that a DSL path exists in the mander.

        If the path does not exist, an assertion error will be raised.

        Parameters
        ----------
        path : str
            The path to check for.
        """
        assert (self.root / path).exists()

    def get_child(self, *pathsegments: str):
        """
        Get a child mander from the current mander.

        Parameters
        ----------
        *pathsegments : str
            The path to the child mander.
        """
        return InfoMander(str(Path(*pathsegments)), root=self.project_path)

    def children(self):
        """Return all direct children of the mander."""
        return [
            InfoMander(str(p.relative_to(self.project_path)), root=self.project_path)
            for p in self.project_path.iterdir()
            if p.is_dir() and (p.stem not in InfoMander.RESERVED_FOLDERS)
        ]

    def get(self, dsl_str):
        """
        Get a property from the mander using the DSL syntax.

        Parameters
        ----------
        dsl_str : str
            The DSL string to use.
        """
        if "@mander" not in dsl_str:
            raise ValueError("@mander is needed at the start of dsl string")
        path = [p for p in dsl_str.replace("@mander", "").split("/") if p]
        # There is no path to another mander, but we may have a nested property
        if len(path) == 1:
            if path[0] == "*":
                return self.children()
            return InfoMander.get_property(self, path[0])
        mander = self.get_child(*path[:-1])
        return mander.get_property(mander, path[-1])

    def __repr__(self):
        """Return a string representation of the mander."""
        return f"InfoMander({self.project_path})"

    def __eq__(self, other):
        """Return True if both instances have the same project path."""
        if isinstance(other, InfoMander):
            return self.project_path == other.project_path
        return False
