import random

import cv2
import numpy as np

import converters

def scale_iterable(iterable, factor):
    return tuple([i * factor for i in iterable])

def offset_iterable(iterable, low, high, minimum=None, maximum=None):
    l = []

    for element in iterable:
        offset_value = element + random.randint(low, high)

        if minimum is not None and offset_value < minimum:
            offset_value = minimum
        elif maximum is not None and offset_value > maximum:
            offset_value = maximum

        l.append(offset_value)

    return tuple(l)

def random_position(problem):
    width = problem.width
    height = problem.height
    return (random.randint(0, width - 1), random.randint(0, height - 1))

def random_color():
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )

def offset_color(color, low=-10, high=10):
    return offset_iterable(color, low, high, 0, 255)

class BaseIndividual(object):
    SIZE_RATIO_MIN = 0.005
    SIZE_RATIO_MAX = 0.01

    def __init__(self, problem, fitness=None, **kwargs):
        self.problem = problem
        self.fitness = fitness

    def create_representation(self):
        raise NotImplementedError()

    def mutate(self):
        raise NotImplementedError()

    def breed_with(self, other):
        raise NotImplementedError()

    def json(self):
        return {"fitness" : self.fitness}


class Ellipse(BaseIndividual):
    def __init__(self, problem, center=None, axes=None, angle=None, color=None,
        startAngle=0, endAngle=360, thickness=-1, **kwargs):
        super(Ellipse, self).__init__(problem, **kwargs)

        if center is None:
            center = random_position(problem)
        else:
            center = tuple(center)

        if axes is None:
            axes = self._random_axes(problem)
        else:
            axes = tuple(axes)

        if angle is None:
            angle = random.randint(0, 360)

        if color is None:
            color = random_color()
        else:
            color = tuple(color)

        self.center = center
        self.axes = axes
        self.angle = angle
        self.color = color
        self.startAngle = startAngle
        self.endAngle = endAngle
        self.thickness = thickness

    def json(self):
        d = super(Ellipse, self).json()
        cls_d = {
            "center" : self.center,
            "axes" : self.axes,
            "angle" : self.angle,
            "color" : self.color,
            "startAngle" : self.startAngle,
            "endAngle" : self.endAngle,
            "thickness" : self.thickness,
        }
        d.update(cls_d)
        return d

    def create_representation(self):
        mask = np.zeros_like(self.problem.image)
        d = {
            "center" : self.center,
            "axes" : self.axes,
            "angle" : self.angle,
            "color" : self.color,
            "startAngle" : self.startAngle,
            "endAngle" : self.endAngle,
            "thickness" : self.thickness,
        }

        cv2.ellipse(mask, **d)
        return mask

    def breed_with(self, other):
        color = converters.crossover_uint_iterables(
            self.color,
            other.color,
            1,
        )
        center = converters.crossover_uint_iterables(
            self.center,
            other.center,
            2,
        )
        axes = converters.crossover_uint_iterables(
            self.axes,
            other.axes,
            2,
        )

        return self.__class__(
            self.problem,
            color=color,
            center=center,
            axes=axes,
        )

    def mutate(self):
        return
        mutation_type = random.randint(1, 3)

        # center
        if mutation_type == 1:
            self.center = self.offset_center()
        # size / axes
        elif mutation_type == 2:
            self.axes = self.offset_axes()
        # color
        elif mutation_type == 3:
            self.color = offset_color(self.color)
        else:
            raise ValueError("should never reach here")

    def offset_center(self, low=-10, high=10):
        return self.offset_iterable(self.center, low, high)

    def offset_axes(self, low=-10, high=10):
        return self.offset_iterable(self.axes, low, high, minimum=5)

    def _random_axes(self, problem):
        width = problem.width
        height = problem.height

        x_min = int(width * self.SIZE_RATIO_MIN)
        x_max = int(width * self.SIZE_RATIO_MAX)

        y_min = int(height * self.SIZE_RATIO_MIN)
        y_max = int(height * self.SIZE_RATIO_MAX)

        return (random.randint(x_min, x_max), random.randint(y_min, y_max))


name_to_obj = {}
to_check = [BaseIndividual]
while to_check:
    cls = to_check.pop()
    for sub_cls in cls.__subclasses__():
        name = sub_cls.__name__
        if name not in name_to_obj:
            name_to_obj[name] = sub_cls
            to_check.append(sub_cls)
