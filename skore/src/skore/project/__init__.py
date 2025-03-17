import sys
import typing

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


if not (PLUGINS := entry_points(group="skore.plugins.project")):
    raise SystemError("No project plugin found, please install at least one.")


def open(
    mode: typing.Literal[tuple(PLUGINS.names)],  # type: ignore[valid-type]
    /,
    *args,
    **kwargs,
):
    if mode not in PLUGINS.names:
        raise ValueError(
            f"Unknown mode '{mode}'. Available modes {', '.join(PLUGINS.names)}."
        )

    return PLUGINS[mode].load()(*args, **kwargs)
