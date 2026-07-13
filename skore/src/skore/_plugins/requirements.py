import sys
import types
from importlib.metadata import packages_distributions

BUILTIN = {"built-in", "frozen"}


def is_builtin(top_level_module: str, module: types.ModuleType) -> bool:
    return (
        (top_level_module in sys.stdlib_module_names) or
        (
            (module.__spec__ is not None) and
            (module.__spec__.origin in BUILTIN)
        )
    )


def infer() -> list[str]:
    module_to_distribution = packages_distributions()
    packages: set[str] = set()

    for name, module in sys.modules.items():
        top_level_module = name.partition(".")[0]

        if is_builtin(top_level_module, module):
            continue

        top_level_module = name.partition(".")[0]
        distribution = module_to_distribution.get(top_level_module)

        if distribution:
            packages.update(distribution)

    return sorted(packages)
