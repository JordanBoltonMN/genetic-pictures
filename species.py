import cv2
import numpy as np

import random


class Individual(object):
    __slots__ = (
        "fitness",
        "center",
        "axes",
        "angle",
        "startAngle",
        "endAngle",
        "color",
        "thickness",
    )

    mask_kwargs = set(__slots__) - {"fitness",}

    SIZE_RATIO_MIN = 0.001
    SIZE_RATIO_MAX = 0.1

    def __init__(self, problem, **kwargs):
        self.problem = problem
        values = self.random_values(problem)
        values.update(kwargs)

        for key in self.__slots__:
            v = values[key]
            if type(v) is list:
                v = tuple(v)
            setattr(self, key, v)

    def create_mask(self, problem):
        raise NotImplementedError()

    def random_values(self, problem):
        raise NotImplementedError()

    def breed_with(self, other):
        raise NotImplementedError()

    def json(self):
        d = {}
        for key in self.__slots__:
            d[key] = getattr(self, key)
        return d


class Ellipse(Individual):
    def create_mask(self, problem):
        mask = np.zeros_like(problem.image)
        d = {}
        for key in self.mask_kwargs:
            d[key] = getattr(self, key)

        cv2.ellipse(mask, **d)
        return mask

    def random_values(self, problem):
        width = problem.width
        height = problem.height

        x_min = int(width * self.SIZE_RATIO_MIN)
        x_max = int(width * self.SIZE_RATIO_MAX)

        y_min = int(height * self.SIZE_RATIO_MIN)
        y_max = int(height * self.SIZE_RATIO_MAX)

        return {
            "center" : (
                random.randint(0, width - 1),
                random.randint(0, height - 1),
            ),
            "axes" : (
                random.randint(x_min, x_max),
                random.randint(y_min, y_max),
            ),
            "angle" : random.randint(0, 360),
            "startAngle" : 0,
            "endAngle" : 360,
            "color" : (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            ),
            "thickness" : -1,
            "fitness" : None
        }

    def breed_with(self, other):
        weight = random.random()
        multiplier1 = lambda x: x * weight
        multiplier2 = lambda x: x * (1 - weight)
        summer = lambda l: int(sum(l))

        def f(iter1, iter2):
            scaled_iter1 = map(multiplier1, iter1)
            scaled_iter2 = map(multiplier2, iter2)
            return tuple(map(summer, zip(scaled_iter1, scaled_iter2)))

        init_kwargs = {
            "center" : f(self.center, other.center),
            "axes" : f(self.axes, other.axes),
            "color" : f(self.color, other.color),
        }
        return cls(self.problem, **init_kwargs)

    def mutate(self):
        pass

name_to_obj = {}
for cls in Individual.__subclasses__():
    name_to_obj[cls.__name__] = cls
