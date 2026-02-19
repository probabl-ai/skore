"""Common ``skore-hub-project`` exceptions."""


class ForbiddenException(PermissionError):
    """Exception raised when attempting an action without adequate permissions."""


class NotFoundException(FileNotFoundError):
    """Exception raised when a requested resource is not found."""
