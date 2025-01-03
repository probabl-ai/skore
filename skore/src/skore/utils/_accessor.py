def _check_supported_ml_task(supported_ml_tasks):
    def check(accessor):
        supported_task = any(
            task in accessor._parent._ml_task for task in supported_ml_tasks
        )

        if not supported_task:
            raise AttributeError(
                f"The {accessor._parent._ml_task} task is not a supported task by "
                f"function called. The supported tasks are {supported_ml_tasks}."
            )

        return True

    return check


def register_accessor(name, target_cls):
    """Register an accessor for a class.

    Parameters
    ----------
    name : str
        The name of the accessor.
    target_cls : type
        The class to register the accessor for.
    """

    def decorator(accessor_cls):
        def getter(self):
            attr = f"_accessor_{accessor_cls.__name__}"
            if not hasattr(self, attr):
                setattr(self, attr, accessor_cls(self))
            return getattr(self, attr)

        setattr(target_cls, name, property(getter))
        return accessor_cls

    return decorator
