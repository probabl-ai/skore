from __future__ import annotations

from enum import StrEnum, auto


class DisplayType(StrEnum):
    ANY = auto()
    ARRAY = auto()
    BOOLEAN = auto()
    DATE = auto()
    DATETIME = auto()
    INTEGER = auto()
    STRING = auto()

    @staticmethod
    def infer():
        raise NotImplementedError
