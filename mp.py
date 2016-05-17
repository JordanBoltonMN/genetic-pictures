from __future__ import division

import math
from multiprocessing import Pool
from itertools import izip

import numpy as np
from PIL import Image

import converters


class MultiprocessingBase(object):
    def __init__(self, processes=3, chunksize=16, rows_per_iter=128):
        self.processes = processes
        self.chunksize = chunksize
        self.rows_per_iter = rows_per_iter

    def __call__(self, package):
        f = self.mapping_function()
        packaged = self.packager(package)
        imap_iter = self.map(f, packaged)
        return self.reduce(imap_iter)

    def json(self):
        return {
            "processes" : self.processes,
            "chunksize" : self.chunksize,
            "rows_per_iter" : self.rows_per_iter,
        }

    def packager(self, package):
        return (i for i in package)

    def mapping_function(self):
        raise NotImplementedError()

    def map(self, f, packaged):
        processes = Pool(processes=self.processes)
        imap_iter = processes.imap(
            f,
            packaged,
            chunksize=self.chunksize,
        )

        processes.close()
        return imap_iter

    def reduce(self, imap_iter):
        return (i for i in imap_iter)


class MPIndividualFitness(MultiprocessingBase):
    def __init__(self, target_np, rows_per_iter=64, **kwargs):
        super(MPIndividualFitness, self).__init__(**kwargs)
        self.target_np = target_np
        (height, width, _) = target_np.shape
        self.target_dimensions = (height, width)
        self.rows_per_iter = rows_per_iter

    def packager(self, package):
        self.inhabitants = package

        for i, individual in enumerate(self.inhabitants):
            representation = individual.representation()
            gen1 = np_slicing_generator(self.target_np, self.rows_per_iter)
            gen2 = np_slicing_generator(representation, self.rows_per_iter)
            for (target_slice, representation_slice) in izip(gen1, gen2):
                yield (target_slice, representation_slice, i)

    def mapping_function(self):
        return image_comparison_rgba

    def reduce(self, imap_iter):
        tally = {}
        for (index, partial_difference) in imap_iter:
            if index not in tally:
                tally[index] = 0

            tally[index] += partial_difference

        for (index, difference) in tally.items():
            self.inhabitants[index].fitness = difference


def np_slicing_generator(np_array, rows_per_iter):
    (height, width, _) = np_array.shape
    row_index = 0

    while row_index < height:
        start = row_index
        stop = start + rows_per_iter

        yield np_array[start:stop]

        row_index += rows_per_iter


def image_comparison_rgba((slice1, slice2, index)):
    # ensure no overflow issues
    slice1 = slice1.astype(dtype=np.int32)
    slice2 = slice2.astype(dtype=np.int32)

    rgb_diff_squared = np.square(slice1 - slice2)
    pixel_diff_squared = np.sum(rgb_diff_squared, axis=-1)
    pixel_diff = np.sqrt(pixel_diff_squared)
    slice_diff = int(np.sum(pixel_diff))
    return (index, slice_diff)
