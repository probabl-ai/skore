from functools import cached_property

import numpy


class NumpyArrayItem:
    def __init__(self, array_list: list, /):
        self.array_list = array_list

    @cached_property
    def array(self) -> numpy.ndarray:
        return numpy.asarray(self.array_list)

    @property
    def __dict__(self):
        return {"array_list": self.array_list}

    @classmethod
    def factory(cls, array: numpy.ndarray) -> NumpyArrayItem:
        instance = cls(array.tolist())

        # add array as cached property
        instance.array = array

        return instance
