from skore.persistence.repository import ItemRepository, ViewRepository
from skore.persistence.storage import InMemoryStorage
from skore.persistence.storage.skore_hub_storage import SkoreHubStorage
from skore.project import Project

if "storage" not in globals():
    storage = SkoreHubStorage("my-project")

project = Project(
    item_repository=ItemRepository(storage=storage),
    view_repository=ViewRepository(storage=InMemoryStorage()),
)

project.put("my-key", 1)
project.put("my-key", 2)
project.put("my-key", 3)
project.put("my-other-key", 3)

print(project.get("my-key"))
print(project.get("my-key", metadata=True))
print(project.get("my-key", version="all"))
print(project.get("my-key", version="all", metadata=True))
print(list(project))
print(project.keys())
