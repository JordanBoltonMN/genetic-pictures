from math import ceil

import numpy as np
from multiprocessing import Pool

import converters

class BaseEvaluator(object):
    def __init__(self, reverse_sort=False, **kwargs):
        self.reverse_sort = reverse_sort

    def __call__(self, image, pool):
        raise NotImplementedError()

    def _fitness_calculator(self, image, representation, **kwargs):
        raise NotImplementedError()

    def json(self):
        return {"reverse_sort" : self.reverse_sort}

class MultiprocessingEvaluator(BaseEvaluator):
    def __init__(self, processes=3, chunksize=8, rows_per_iter=10, **kwargs):
        super(MultiprocessingEvaluator, self).__init__(**kwargs)
        self.processes = processes
        self.chunksize = chunksize
        self.rows_per_iter = rows_per_iter

    def json(self):
        result = super(MultiprocessingEvaluator, self).json()
        result.update(processes=self.processes, chunksize=self.chunksize)
        return result

    def __call__(self, image, pool):
        f = self._imap_function()
        packaged = self._imap_packager(image, pool)
        imap_iter = self._imap(f, packaged)
        return self._imap_reconstructor(image, pool, imap_iter)

    def _imap_function(self):
        return sliced_fitness

    def _imap_packager(self, image, pool):
        return sliced_image_generator(
            image,
            pool,
            self,
            self.rows_per_iter,
        )

    def _imap(self, f, packaged):
        processes = Pool(processes=self.processes)
        imap_iter = processes.imap(
            f,
            packaged,
            chunksize=self.chunksize,
        )

        processes.close()
        return imap_iter

    def _imap_reconstructor(self, image, pool, imap_iter):
        calculated_fitness = {}

        for (individual_index, fitness) in imap_iter:
            if individual_index not in calculated_fitness:
                calculated_fitness[individual_index] = 0
            calculated_fitness[individual_index] += fitness

        for individual_index, fitness in calculated_fitness.items():
            pool[individual_index].fitness = fitness

        return pool

class ColorDifference(MultiprocessingEvaluator):
    def _fitness_calculator(self, image, representation, **kwargs):
        fitness = 0
        weight = kwargs.get("weight", None)

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
        raise NotImplementedError()

class RGBDifference(ColorDifference):
    def rgb_fitness(self, rgb1, rgb2):
        """ sum the difference between color octals """
        return sum(map(lambda (a, b): abs(int(a) - int(b)), zip(rgb1, rgb2)))

class WeightedRGBDifference(RGBDifference):
    def _imap_packager(self, image, pool):
        weight = weight_matrix(image, pool, self.processes, self.chunksize)

        def dict_updater(d):
            start = d["start"]
            stop = d["stop"]
            return {"calculator_kwargs" : {"weight" : weight[start:stop]}}

        return sliced_image_generator(
            image,
            pool,
            self,
            self.rows_per_iter,
            dict_updater,
        )


class CIELabDifference(ColorDifference):
    def rgb_fitness(self, rgb1, rgb2):
        """ sum the difference between color octals """
        return converters.cielab_distance_from_rgb(rgb1, rgb2)


class WeightedCIELabDifference(RGBDifference):
    def _imap_packager(self, image, pool):
        weight = weight_matrix(image, pool, self.processes, self.chunksize)

        def dict_updater(d):
            start = d["start"]
            stop = d["stop"]
            return {"calculator_kwargs" : {"weight" : weight[start:stop]}}

        return sliced_image_generator(
            image,
            pool,
            self,
            self.rows_per_iter,
            dict_updater,
        )


def sliced_image_generator(image, pool, evaluator, rows_per_iter,
                           dict_updater=None):
    def dict_builder(individual_index, representation, start, stop):
        return {
            "evaluator" : evaluator,

            "image_slice" : image[start:stop],
            "row_slice" : representation[start:stop],

            "individual_index" : individual_index,

            "start" : start,
            "stop" : stop,
        }

    def generator():
        for individual_index, individual in enumerate(pool):
            representation = individual.create_representation()
            height = representation.shape[0]
            row_index = 0

            while row_index < height:
                start = row_index
                stop = start + rows_per_iter

                d = dict_builder(individual_index, representation, start, stop)
                if dict_updater is not None:
                    d.update(dict_updater(d))

                yield d

                row_index += rows_per_iter

    return generator()


def sliced_fitness(d):
    individual_index = d["individual_index"]
    image_slice = d["image_slice"]
    row_slice = d["row_slice"]

    evaluator = d["evaluator"]
    calculator_kwargs = d.get("calculator_kwargs", {})

    fitness = evaluator._fitness_calculator(
        image_slice,
        row_slice,
        **calculator_kwargs
    )

    return (individual_index, fitness)


def weight_matrix(image, pool, processes, chunksize):
    distribution = distribution_matrix(pool, processes, chunksize)
    tmp = distribution.astype(np.float32)
    with np.errstate(divide='ignore'):
        weight = np.where(tmp > 0, np.reciprocal(tmp), 0)

    return weight


def distribution_matrix(pool, processes, chunksize):
    processes = Pool(processes=processes)
    imap_iter = processes.imap(
        individual_pixels,
        pool,
        chunksize=chunksize,
    )
    result = sum(imap_iter)
    processes.close()
    return result


def individual_pixels(individual):
    representation = individual.create_representation()
    pixel_locations = np.where(np.any(representation, axis=-1), 1, 0)
    # will be summing later,
    # conversion prevents overflow on 255 or more overlapping representations
    pixel_locations.astype(np.int32, copy=False)
    return pixel_locations


name_to_obj = {}
to_check = [BaseEvaluator]
while to_check:
    cls = to_check.pop()
    for sub_cls in cls.__subclasses__():
        name = sub_cls.__name__
        if name not in name_to_obj:
            name_to_obj[name] = sub_cls
            to_check.append(sub_cls)
