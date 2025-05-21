"""Alias top level function and class of the project submodule."""

from skore.externals._pandas_accessors import _register_accessor
from skore.project.project import Project
from skore.project.reports import _ReportsAccessor

_register_accessor("reports", Project)(_ReportsAccessor)

__all__ = ["Project"]
