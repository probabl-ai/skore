from skore.persistence.repository import ItemRepository, ViewRepository
from skore.persistence.storage import InMemoryStorage
from skore.persistence.storage.skore_hub_storage import SkoreHubStorage
from skore.project import Project

domain1 = "https://skh.k.probabl.dev"
domain2 = "http://0.0.0.0:8000"
domain = domain1

storage = SkoreHubStorage.from_project_name("my-project", domain=domain)
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

my_project_id = storage.project_id

del storage
del project

storage = SkoreHubStorage(project_id=my_project_id, domain=domain)
project = Project(
    item_repository=ItemRepository(storage=storage),
    view_repository=ViewRepository(storage=InMemoryStorage()),
)

print(project.get("my-key"))
print(project.get("my-key", metadata=True))
print(project.get("my-key", version="all"))
print(project.get("my-key", version="all", metadata=True))
print(list(project))
print(project.keys())

# ---

from sklearn import datasets, linear_model

from skore import CrossValidationReporter

X, y = datasets.load_diabetes(return_X_y=True)
lasso = linear_model.Lasso()
reporter = CrossValidationReporter(lasso, X, y, cv=3)

project.put("reporter", reporter)

print(reporter.cv_results)
print(project.get("reporter").cv_results)
