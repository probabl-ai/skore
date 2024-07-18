import json
import pathlib

__all__ = ["v0"]


with open(pathlib.Path(__file__).parent / "v0.json") as f:
    v0 = json.load(f)
