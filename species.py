import random

import numpy as np
from PIL import Image, ImageDraw

import converters


def random_point(dimensions):
    (height, width) = dimensions
    return [
        random.randint(0, width - 1),
        random.randint(0, height - 1),
    ]


def random_rgba():
    return [
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    ]


class Individual(object):
    def __init__(self, image_dimensions, fitness=None):
        self.fitness = fitness
        self.image_dimensions = image_dimensions

    def mate(self, other):
        raise NotImplementedError()

    def mutate(self, mutation_rate):
        raise NotImplementedError()

    def representation(self):
        raise NotImplementedError()

    def json(self):
        return {"fitness" : self.fitness}


class TrianglePool(Individual):
    # rgba + 3 (x, y) points
    POOL_DATA = 4 + (3 * 2)

    def __init__(self, image_dimensions, size=128, pool=None,
                 **kwargs):
        super(TrianglePool, self).__init__(image_dimensions, **kwargs)

        # pool is a 2D array in the form of
        # [individial][r, g, b, a, x1, y1, x2, y2, x3, y3]

        if pool is not None:
            self.pool = self.load_pool(pool)
        else:
            self.pool = self.create_pool(size)

    def load_pool(self, pool):
        """
            initializes the pool attribute to the list of lists
        """
        pool_size = len(pool)
        return np.array(pool, dtype=np.uint8) \
                 .reshape((pool_size, self.POOL_DATA))

    def create_pool(self, size):
        pool = []
        for i in range(size):
            color = random_rgba()
            points = self.random_triangle(self.image_dimensions)
            pool.append(color + points)

        return self.load_pool(pool)

    def mate(self, other):
        """
            returns an instance of the same class.
            The child's pool is a combination of both parents.
            Crossover is done by converting both pools to bits, then
            splitting somewhere between 20-80 percent.
        """
        pa_bits = np.unpackbits(self.pool.view(np.uint8))
        ma_bits = np.unpackbits(other.pool.view(np.uint8))

        assert pa_bits.size == ma_bits.size

        split = int(random.uniform(0.20, 0.80) * pa_bits.size)
        child_bits = np.zeros_like(pa_bits)
        child_bits[:split] = pa_bits[:split]
        child_bits[split:] = ma_bits[split:]

        child = np.packbits(child_bits) \
                  .view(self.pool.dtype) \
                  .reshape(self.pool.shape)

        return self.__class__(self.image_dimensions, pool=child)

    def mutate(self, mutation_rate):
        """
            converts pool to a bit array, then randomly flips some bits based
            on the given mutation rate.
        """

        bits = np.unpackbits(self.pool.view(np.uint8))
        mutations = np.random.random(bits.size)
        mutations_location = mutations <= mutation_rate

        # I don't know how to 'flip 0 to 1 and 1 to 0 where mask exists'
        # so this is a ghetto solution
        bits[mutations_location] += 2
        bits[bits == 2] = 1
        bits[bits == 3] = 0

        self.pool = np.packbits(bits) \
                      .view(self.pool.dtype) \
                      .reshape(self.pool.shape)

    def representation(self):
        (height, width) = self.image_dimensions
        result = Image.new("RGBA", (width, height), (255, 255, 255, 0))

        for individual in self.pool:
            rgba = tuple(individual[:4])

            points = [(individual[i], individual[i + 1]) \
                      for i in range(4, len(individual), 2)]

            workspace = Image.new("RGBA", (width, height), (255, 255, 255, 0))
            drawer = ImageDraw.Draw(workspace)
            drawer.polygon(points, fill=rgba)

            result = Image.alpha_composite(result, workspace)

        return np.array(result, dtype=np.uint8)

    def json(self):
        d = super(TrianglePool, self).json()
        d.update(pool=self.pool.tolist())
        return d

    def random_triangle(self, dimensions):
        points = []
        for _ in range(3):
            points.extend(random_point(dimensions))
        return points
