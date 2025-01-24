from skore.persistence.repository import ItemRepository, ViewRepository
from skore.persistence.storage import InMemoryStorage
from skore.persistence.storage.skore_hub_storage import SkoreHubStorage
from skore.project import Project

if "storage" not in globals():
    storage = SkoreHubStorage("my-project")

storage["my-key"] = {
    "item_class_name": "<item-class-name>",
    "item": {},
}

project = Project(
    item_repository=ItemRepository(storage=storage),
    view_repository=ViewRepository(storage=InMemoryStorage())
)

project.put("my-key", 1)
print(project.get("my-key"))
