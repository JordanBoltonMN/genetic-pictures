from math import ceil

import numpy as np
from multiprocessing import Pool


class BaseEvaluator(object):
    JSON_KEYS = {"reverse_sort", "processes"}

    def __init__(self, **kwargs):
        init_kwargs = self.default_kwargs()
        init_kwargs.update(**kwargs)

        self.reverse_sort = init_kwargs["reverse_sort"]
        self.processes = init_kwargs["processes"]
        self.chunksize = init_kwargs["chunksize"]

    def default_kwargs(self):
        return {
            "reverse_sort" : True,
            "processes" : 2,
            "chunksize" : 8,
        }

    def json(self):
        return {
            "reverse_sort" : self.reverse_sort,
            "processes" : self.processes,
            "chunksize" : self.chunksize,
        }

    def __call__(self, image, pool):
        f = self._imap_function()
        packaged = self._imap_packager(image, pool)
        return self._imap(image, f, packaged)

    def _imap_function(self):
        raise NotImplementedError()

    @classmethod
    def _fitness_calculator(cls, image, representation, weight=None):
        raise NotImplementedError()

    def _imap_packager(self, image, pool):
        return ((image, i, self) for i in pool)

    def _imap(self, image, f, packaged):
        processes = Pool(processes=self.processes)
        imap_iter = processes.imap(
            f,
            packaged,
            chunksize=self.chunksize,
        )

        result = [individual for individual in imap_iter]
        processes.close()

        return result

class RGBDifference(BaseEvaluator):
    def _imap_function(self):
        return update_individual_fitness


    def _fitness_calculator(self, image, representation, weight=None):
        fitness = 0

        overlap = np.bitwise_and(image, representation)
        for (y, x) in np.argwhere(overlap.any(axis=-1)):
            rgb1 = image[y][x]
            rgb2 = representation[y][x]
            pixel_fitness = self.rgb_fitness(rgb1, rgb2)

            # assumed to be a numpy array
            if weight is not None:
                pixel_fitness = int(ceil(pixel_fitness * weight[y][x]))

            fitness += pixel_fitness

        return fitness

    def rgb_fitness(self, rgb1, rgb2):
        """ sum the difference between color octals """
        return sum(map(lambda (a, b): abs(int(a) - int(b)), zip(rgb1, rgb2)))


class WeightedRGBDifference(RGBDifference):
    def _imap_function(self):
        return weighted_update_individual_fitness

    def _imap_packager(self, image, pool):
        weight = weight_matrix(
            image,
            pool,
            self.processes,
            self.chunksize
        )
        return ((image, i, self, weight) for i in pool)


def update_individual_fitness((image, individual, evaluator)):
    """
    used by multiprocessing.Pool.map, which can only take a single value,
    but the function needs two arguments which is why the syntax is weird
    """
    representation = individual.create_representation()
    fitness = evaluator._fitness_calculator(image, representation)
    individual.fitness = fitness
    return individual


def weighted_update_individual_fitness((image, individual, evaluator, weight)):
    """
    used by multiprocessing.Pool.map, which can only take a single value,
    but the function needs two arguments which is why the syntax is weird
    """
    representation = individual.create_representation()
    fitness = evaluator._fitness_calculator(image, representation, weight)
    individual.fitness = fitness
    return individual


def individual_pixels(individual):
    representation = individual.create_representation()
    pixel_locations = np.where(np.any(representation, axis=-1), 1, 0)
    # will be summing later,
    # conversion prevents overflow on 255 or more overlapping representations
    pixel_locations.astype(np.int32, copy=False)
    return pixel_locations


def distribution_matrix(image, pool, processes, chunksize):
    processes = Pool(processes=processes)
    imap_iter = processes.imap(
        individual_pixels,
        pool,
        chunksize=chunksize,
    )
    result = sum(imap_iter)
    processes.close()
    return result


def weight_matrix(image, pool, processes, chunksize):
    distribution = distribution_matrix(image, pool, processes, chunksize)
    tmp = distribution.astype(np.float32)
    with np.errstate(divide='ignore'):
        weight = np.where(tmp > 0, np.reciprocal(tmp), 0)

    return weight


name_to_obj = {}
to_check = [BaseEvaluator]
while to_check:
    cls = to_check.pop()
    for sub_cls in cls.__subclasses__():
        name = sub_cls.__name__
        if name not in name_to_obj:
            name_to_obj[name] = sub_cls
            to_check.append(sub_cls)
