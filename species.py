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


def random_x_position(width):
    return random.randint(0, width - 1)


def random_y_position(height):
    return random.randint(0, height - 1)


def random_position(image_dimensions):
    (width, height) = image_dimensions
    return (random_x_position(width), random_y_position(height))


def random_color():
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )


def offset_color(color, low=-10, high=10):
    return offset_iterable(color, low, high, 0, 255)


class BaseIndividual(object):
    def __init__(self, problem, fitness=None):
        (height, width, _) = problem.image.shape

        self.image_dimensions = (height, width)
        self.image_dtype = problem.image.dtype
        self.problem = problem
        self.fitness = fitness

    def create_representation(self, alpha=None):
        raise NotImplementedError()

    def mutate(self):
        raise NotImplementedError()

    def json(self):
        return {"fitness" : self.fitness}


class Polygon(BaseIndividual):
    def __init__(self, problem, num_points, points=None, color=None, alpha=200,
                 **kwargs):
        super(Polygon, self).__init__(problem, **kwargs)

        if points is None:
            points = []
            for _ in range(num_points):
                points.append(random_position(self.image_dimensions))
            # points = [random_position(problem) for _ in range(num_points)]
            points = np.array(points, dtype=np.int32)
            points = points.reshape(-1, 1, 2)
        else:
            assert len(points) == num_points
            points = np.array(points)

        if color is None:
            color = random_color()

        self.num_points = num_points
        self.points = points
        self.color = random_color()
        self.alpha = alpha

    def create_representation(self, alpha=None):
        (height, width) = self.image_dimensions
        dtype = self.image_dtype

        representation = np.full((height, width, 3), 0, dtype=dtype)
        cv2.fillPoly(representation, [self.points], self.color)

        if alpha is None:
            return representation

        alpha_mask = np.full((height, width, 1), 0, dtype=dtype)
        alpha_mask[np.any(representation, axis=-1)] = alpha
        return np.dstack((representation, alpha_mask))

    def json(self):
        d = super(Polygon, self).json()
        d.update(
            num_points=self.num_points,
            points=self.points.tolist(),
            color=self.color,
            alpha=self.alpha,
        )
        return d

    def mutate(self):
        base = self.json()

        mutation_type = random.randint(0, 1)
        if mutation_type is 0:
            return self._mutate_point()
        elif mutation_type is 1:
            return self._mutate_color()
        else:
            raise ValueError("should never reach here")

    def _mutate_point(self):
        kwargs = self.json()
        points = kwargs["points"]
        point_index = random.randint(0, self.num_points - 1)
        (height, width) = self.image_dimensions

        # [True, False] == [y coordinate, x_coordinate]
        if random.choice([True, False]):
            points[point_index][0][0] = random_y_position(height)
        else:
            points[point_index][0][1] = random_x_position(width)

        return kwargs

    def _mutate_color(self):
        kwargs = self.json()

        color = list(kwargs["color"])
        color[random.randint(0, 2)] = random.randint(0, 255)
        kwargs["color"] = tuple(color)

        return kwargs


class Triangle(Polygon):
    def __init__(self, problem, **kwargs):
        kwargs["num_points"] = 3
        super(Triangle, self).__init__(problem, **kwargs)


name_to_obj = {}
to_check = [BaseIndividual]
while to_check:
    cls = to_check.pop()
    for sub_cls in cls.__subclasses__():
        name = sub_cls.__name__
        if name not in name_to_obj:
            name_to_obj[name] = sub_cls
            to_check.append(sub_cls)
