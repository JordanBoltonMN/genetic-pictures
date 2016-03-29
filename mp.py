from multiprocessing import Pool
from itertools import izip

import numpy as np
from PIL import Image

class MultiprocessingBase(object):
    def __init__(self, processes=3, chunksize=16, rows_per_iter=128):
        self.processes = processes
        self.chunksize = chunksize
        self.rows_per_iter = rows_per_iter

    def __call__(self, package):
        packaged = self._imap_packager(package)
        f = self._imap_function()
        imap_iter = self._imap(f, packaged)
        return self._imap_reconstructor(imap_iter)

    def json(self):
        return {
            "processes" : "self.processes",
            "chunksize" : "self.chunksize",
            "rows_per_iter" : "self.rows_per_iter",
        }

    def _imap_packager(self, package):
        return iter(package)

    def _imap_function(self):
        raise NotImplementedError()

    def _imap(self, f, packaged):
        processes = Pool(processes=self.processes)
        imap_iter = processes.imap(
            f,
            packaged,
            chunksize=self.chunksize,
        )

        processes.close()
        return imap_iter

    def _imap_reconstructor(self, imap_iter):
        return (i for i in imap_iter)


class MPWorldRepresentation(MultiprocessingBase):
    def __init__(self, image_dimensions, **kwargs):
        super(MPWorldRepresentation, self).__init__(**kwargs)

        (height, width) = image_dimensions
        base = np.full((height, width, 4), 0, dtype=np.uint8)
        self.base = Image.fromarray(base)

    def _imap_packager(self, package):
        return generator(package)

    def _imap_function(self):
        return create_alpha_representation

    def _imap_reconstructor(self, imap_iter):
        base = self.base

        for np_individual_representation in imap_iter:
            rep = Image.fromarray(np_individual_representation)
            base.paste(rep, (0, 0), rep)

        return base


class MPWorldFitness(MultiprocessingBase):
    def __init__(self, rows_per_iter=16, **kwargs):
        super(MPWorldFitness, self).__init__(**kwargs)
        self.rows_per_iter = rows_per_iter

    def _imap_packager(self, package):
        (target, representation) = package
        gen1 = image_slicing_generator(target, self.rows_per_iter)
        gen2 = image_slicing_generator(representation, self.rows_per_iter)
        return izip(gen1, gen2)

    def _imap_function(self):
        return image_comparison_rgb

    def _imap_reconstructor(self, imap_iter):
        return sum(imap_iter)


def generator(iterable):
    for i in iterable:
        yield i


def image_slicing_generator(image, rows_per_iter):
    (height, width, _) = image.shape
    row_index = 0

    while row_index < height:
        start = row_index
        stop = start + rows_per_iter

        yield image[start:stop].flatten()

        row_index += rows_per_iter


def image_comparison_rgb((slice1, slice2)):
    diff = np.subtract(slice1, slice2)
    abs_diff = np.absolute(diff)
    return np.sum(abs_diff)


def create_alpha_representation(individual):
    return individual.create_representation(alpha=individual.alpha)
