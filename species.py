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
        "problem",
    )

    MASK_KEYS = set(__slots__) - {"fitness", "problem"}
    JSON_KEYS = set(__slots__) - {"problem",}

    SIZE_RATIO_MIN = 0.01
    SIZE_RATIO_MAX = 0.1

    def __init__(self, problem, **kwargs):
        self.problem = problem
        self.fitness = None

        values = self.random_values()
        values.update(kwargs)

        for key in self.MASK_KEYS:
            v = values[key]
            if type(v) is list:
                v = tuple(v)
            setattr(self, key, v)

    def create_representation(self):
        raise NotImplementedError()

    def random_values(self):
        raise NotImplementedError()

    def breed_with(self, other):
        raise NotImplementedError()

    def json(self):
        d = {}
        for key in self.JSON_KEYS:
            d[key] = getattr(self, key)
        return d

    def offset_iterable(self, iterable, low, high, minimum=None, maximum=None):
        l = []

        for element in iterable:
            offset_value = element + random.randint(low, high)

            if minimum is not None and offset_value < minimum:
                offset_value = minimum
            elif maximum is not None and offset_value > maximum:
                offset_value = maximum

            l.append(offset_value)

        return tuple(l)

    def offset_color(self, low=-10, high=10):
        return self.offset_iterable(self.color, low, high, 0, 255)

    def scale_iterable(self, iterable, factor):
        return tuple([i * factor for i in iterable])

    def _random_coordinate(self):
        width = self.problem.width
        height = self.problem.height
        return (random.randint(0, width - 1), random.randint(0, height - 1))

    def _random_color(self):
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )


class Ellipse(Individual):
    def create_representation(self):
        mask = np.zeros_like(self.problem.image)
        d = {}
        for key in self.MASK_KEYS:
            d[key] = getattr(self, key)

        cv2.ellipse(mask, **d)
        return mask

    def random_values(self):
        return {
            "center" : self._random_coordinate(),
            "axes" : self._random_axes(),
            "angle" : random.randint(0, 360),
            "startAngle" : 0,
            "endAngle" : 360,
            "color" : self._random_color(),
            "thickness" : -1,
            "fitness" : None,
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
        mutation_type = random.randint(1, 3)

        # center
        if mutation_type == 1:
            self.center = self.offset_center()
        # size / axes
        elif mutation_type == 2:
            self.axes = self.offset_axes()
        # color
        elif mutation_type == 3:
            self.color = self.offset_color()
        else:
            raise ValueError("should never reach here")

    def offset_center(self, low=-10, high=10):
        return self.offset_iterable(self.center, low, high)

    def offset_axes(self, low=-10, high=10):
        return self.offset_iterable(self.axes, low, high, minimum=5)

    def _random_axes(self):
        width = self.problem.width
        height = self.problem.height

        x_min = int(width * self.SIZE_RATIO_MIN)
        x_max = int(width * self.SIZE_RATIO_MAX)

        y_min = int(height * self.SIZE_RATIO_MIN)
        y_max = int(height * self.SIZE_RATIO_MAX)

        return (random.randint(x_min, x_max), random.randint(y_min, y_max))


name_to_obj = {}
for cls in Individual.__subclasses__():
    name_to_obj[cls.__name__] = cls
