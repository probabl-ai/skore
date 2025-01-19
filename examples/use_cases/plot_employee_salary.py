# %%
import tempfile
from pathlib import Path

temp_dir = tempfile.TemporaryDirectory(prefix="skore_example_")
temp_dir_path = Path(temp_dir.name)

# %%
import skore

my_project = skore.open(temp_dir_path / "my_project")

# %%
# FIXME: add a comment to start the UI at the right location
