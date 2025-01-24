from skore.persistence.storage.skore_hub_storage import SkoreHubStorage

if "storage" not in globals():
    storage = SkoreHubStorage("my-project")

storage["my-key"] = {
    "item_class_name": "<item-class-name>",
    "item": {},
}

storage["my-key"]
