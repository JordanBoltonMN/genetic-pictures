from math import ceil

import numpy as np
from multiprocessing import Pool

import converters

class BaseEvaluator(object):
    def __init__(self, reverse_sort=False, **kwargs):
        self.reverse_sort = reverse_sort

    def __call__(self, image, inhabitants):
        raise NotImplementedError()

    def sort(self, iterable):
        iterable.sort(
            key=lambda individual: individual.fitness,
            reverse=self.reverse_sort,
        )

    def fitness(self, image, representation, **kwargs):
        raise NotImplementedError()

    def json(self):
        return {"reverse_sort" : self.reverse_sort}

class MultiprocessingEvaluator(BaseEvaluator):
    def __init__(self, processes=3, chunksize=8, rows_per_iter=32, **kwargs):
        super(MultiprocessingEvaluator, self).__init__(**kwargs)
        self.processes = processes
        self.chunksize = chunksize
        self.rows_per_iter = rows_per_iter

    def __call__(self, image, inhabitants):
        f = self._imap_function()
        packaged = self._imap_packager(image, inhabitants)
        imap_iter = self._imap(f, packaged)
        inhabitants = self._imap_reconstructor(image, inhabitants, imap_iter)

        self.sort(inhabitants)
        return inhabitants

    def json(self):
        result = super(MultiprocessingEvaluator, self).json()
        result.update(
            processes=self.processes,
            chunksize=self.chunksize,
            rows_per_iter=self.rows_per_iter,
        )
        return result

    def _imap_function(self):
        return sliced_fitness

    def _imap_packager(self, image, inhabitants):
        return sliced_image_generator(
            image,
            inhabitants,
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

    def _imap_reconstructor(self, image, inhabitants, imap_iter):
        calculated_fitness = {}

        for (collection_i, individual_i, fitness) in imap_iter:
            key = (collection_i, individual_i)
            if key not in calculated_fitness:
                calculated_fitness[key] = 0

            calculated_fitness[key] += fitness

        items = calculated_fitness.items()
        for ((collection_i, individual_i), summed_fitness) in items:
            inhabitants[collection_i][individual_i].fitness = summed_fitness

        return inhabitants

class ColorDifference(MultiprocessingEvaluator):
    def fitness(self, image, representation, **kwargs):
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

# class WeightedRGBDifference(RGBDifference):
#     def _imap_packager(self, image, inhabitants):
#         weight = weight_matrix(image, inhabitants, self.processes, self.chunksize)

#         def dict_updater(d):
#             start = d["start"]
#             stop = d["stop"]
#             return {"calculator_kwargs" : {"weight" : weight[start:stop]}}

#         return sliced_image_generator(
#             image,
#             inhabitants,
#             self,
#             self.rows_per_iter,
#             dict_updater,
#         )


class CIELabDifference(ColorDifference):
    def rgb_fitness(self, rgb1, rgb2):
        """ sum the difference between color octals """
        return converters.cielab_distance_from_rgb(rgb1, rgb2)


# class WeightedCIELabDifference(RGBDifference):
#     def _imap_packager(self, image, inhabitants):
#         weight = weight_matrix(
#             image,
#             inhabitants,
#             self.processes,
#             self.chunksize
#         )

#         def dict_updater(d):
#             start = d["start"]
#             stop = d["stop"]
#             return {"calculator_kwargs" : {"weight" : weight[start:stop]}}

#         return sliced_image_generator(
#             image,
#             inhabitants,
#             self,
#             self.rows_per_iter,
#             dict_updater,
#         )


def sliced_image_generator(image, inhabitants, evaluator, rows_per_iter,
                           dict_updater=None):
    def dict_builder(c_index, i_index, representation, start, stop):
        return {
            "evaluator" : evaluator,

            "image_slice" : image[start:stop],
            "row_slice" : representation[start:stop],

            "collection_index" : c_index,
            "individual_index" : i_index,

            "representation" : representation,

            "start" : start,
            "stop" : stop,
        }

    def generator():
        for collection_index, collection in enumerate(inhabitants):
            for individual_index, individual in enumerate(collection):
                representation = individual.create_representation()
                height = representation.shape[0]
                row_index = 0

                while row_index < height:
                    start = row_index
                    stop = start + rows_per_iter

                    d = dict_builder(
                        collection_index,
                        individual_index,
                        representation,
                        start,
                        stop
                    )
                    if dict_updater is not None:
                        d.update(dict_updater(d))

                    yield d

                    row_index += rows_per_iter


    return generator()


def sliced_fitness(d):
    collection_index = d["collection_index"]
    individual_index = d["individual_index"]

    image_slice = d["image_slice"]
    row_slice = d["row_slice"]

    evaluator = d["evaluator"]
    calculator_kwargs = d.get("calculator_kwargs", {})

    fitness = evaluator.fitness(
        image_slice,
        row_slice,
        **calculator_kwargs
    )

    return (collection_index, individual_index, fitness)


# def weight_matrix(image, inhabitants, processes, chunksize):
#     distribution = distribution_matrix(inhabitants, processes, chunksize)
#     tmp = distribution.astype(np.float32)
#     with np.errstate(divide='ignore'):
#         weight = np.where(tmp > 0, np.reciprocal(tmp), 0)

#     return weight


# def distribution_matrix(inhabitants, processes, chunksize):
#     processes = Pool(processes=processes)
#     imap_iter = processes.imap(
#         individual_pixels,
#         inhabitants,
#         chunksize=chunksize,
#     )
#     result = sum(imap_iter)
#     processes.close()
#     return result


# def individual_pixels(individual):
#     representation = individual.create_representation()
#     pixel_locations = np.where(np.any(representation, axis=-1), 1, 0)
#     # will be summing later,
#     # conversion prevents overflow on 255 or more overlapping representations
#     pixel_locations.astype(np.int32, copy=False)
#     return pixel_locations


name_to_obj = {}
to_check = [BaseEvaluator]
while to_check:
    cls = to_check.pop()
    for sub_cls in cls.__subclasses__():
        name = sub_cls.__name__
        if name not in name_to_obj:
            name_to_obj[name] = sub_cls
            to_check.append(sub_cls)
