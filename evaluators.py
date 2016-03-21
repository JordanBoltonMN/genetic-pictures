from math import ceil

import numpy as np
from multiprocessing import Pool

class BaseEvaluator(object):
    def __init__(self, reverse_sort=False, **kwargs):
        self.reverse_sort = reverse_sort

    def __call__(self, image, pool):
        raise NotImplementedError()

    def json(self):
        return {"reverse_sort" : self.reverse_sort}

class MultiprocessingEvaluator(BaseEvaluator):
    def __init__(self, processes=2, chunksize=8, **kwargs):
        super(MultiprocessingEvaluator, self).__init__(**kwargs)
        self.processes = processes
        self.chunksize = chunksize

    def json(self):
        result = super(MultiprocessingEvaluator, self).json()
        result.update({
            "processes" : self.processes,
            "chunksize" : self.chunksize,
        })
        return result

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
        # image, individual, evaluator, weight
        return ((image, i, self, None) for i in pool)

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

class RGBDifference(MultiprocessingEvaluator):
    def _imap_function(self):
        return update_individual_fitness

    def _imap_packager(self, image, pool):
        def helper(individual):
            return {
                "image" : image,
                "evaluator" : self,
                "individual" : individual
            }
        return (helper(individual) for individual in pool)

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
    def _imap_packager(self, image, pool):
        weight = weight_matrix(
            image,
            pool,
            self.processes,
            self.chunksize
        )
        def helper(individual):
            return {
                "image" : image,
                "evaluator" : self,
                "individual" : individual,
                "calculator_kwargs" : {"weight" : weight}
            }
        return (helper(individual) for individual in pool)

def update_individual_fitness(d):
    image = d["image"]
    individual = d["individual"]
    evaluator = d["evaluator"]
    calculator_kwargs = d.get("calculator_kwargs", {})

    representation = individual.create_representation()
    individual.fitness = evaluator._fitness_calculator(
        image,
        representation,
        **calculator_kwargs
    )
    return individual


def weight_matrix(image, pool, processes, chunksize):
    distribution = distribution_matrix(image, pool, processes, chunksize)
    tmp = distribution.astype(np.float32)
    with np.errstate(divide='ignore'):
        weight = np.where(tmp > 0, np.reciprocal(tmp), 0)

    return weight



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


def individual_pixels(individual):
    representation = individual.create_representation()
    pixel_locations = np.where(np.any(representation, axis=-1), 1, 0)
    # will be summing later,
    # conversion prevents overflow on 255 or more overlapping representations
    pixel_locations.astype(np.int32, copy=False)
    return pixel_locations


def rgb_to_xyz(rgb):
    # Observer = 2nd degree, Illuminant = D65
    def helper(octal):
        octal /= 255

        if octal > 0.04045:
            numer = octal + 0.055
            denom = 1.055
            quotient = numer / denom
            octal = quotient ** 2.4
        else:
            octal /= 12.92

        octal *= 100

        return octal

    var_r, var_g, var_b = map(helper, rgb)

    x = (var_r * 0.4124) + (var_g * 0.3576) + (var_b * 0.1805)
    y = (var_r * 0.2126) + (var_g * 0.7152) + (var_b * 0.0722)
    z = (var_r * 0.0193) + (var_g * 0.1192) + (var_b * 0.9505)

    return (x, y, z)

def xyz_to_cielab(xyz, reference=None):
    # Observer = 2nd degree, Illuminant = D65
    if reference is None:
        reference = (95.047, 100.000, 108.883)

    def helper((octal, reference)):
        octal /= reference

        if octal > 0.0085856:
            octal = octal ** (1/3)
        else:
            octal = (octal * 7.787) + (16 / 116)

        return octal

    var_x, var_y, var_z = map(helper, zip(xyz, reference))

    l = (116 * var_y) - 16
    a = 500 * (var_x - var_y)
    b = 200 * (var_y - var_z)

    return (l, a, b)


def rgb_to_cielab(rgb):
    return xyz_to_cielab(rgb_to_xyz(rgb))


def lab_distance_from_rgb(rgb1, rgb2):
    lab1 = rgb_to_cielab(rgb1)
    lab2 = rgb_to_cielab(rgb2)
    return delta_e1994(lab1, lab2)


def delta_e1994(lab1, lab2, weights=None):
    if weights is None:
        weights = (1.0, 1.0, 1.0)

    l1, a1, b1 = lab1
    l2, a2, b2 = lab2

    xC1 = sqrt(a1 ** 2 + b1 ** 2)
    xC2 = sqrt(a2 ** 2 + b2 ** 2)

    xDL = l2 - l1
    xDC = xC2 - xC1
    xDE = distance(lab1, lab2)

    if sqrt(xDE) > ( sqrt(abs(xDL)) + sqrt(abs(xDC)) ):
        xDH = sqrt( (xDE ** 2) - (xDL ** 2) - (xDC ** 2) )
    else:
        xDH = 0

    xSC = 1 + (0.045 * xC1)
    xSH = 1 + (0.015 * xC1)

    xDL /= weights[0]
    xDC /= weights[1] * xSC
    xDH /= weights[2] * xSH

    return sqrt( (xDL ** 2) + (xDC ** 2) + (xDH ** 2) )


def distance(coord1, coord2):
    assert len(coord1) == len(coord2)
    arr1 = np.array(coord1)
    arr2 = np.array(coord2)
    return np.linalg.norm(arr1 - arr2)


name_to_obj = {}
to_check = [BaseEvaluator]
while to_check:
    cls = to_check.pop()
    for sub_cls in cls.__subclasses__():
        name = sub_cls.__name__
        if name not in name_to_obj:
            name_to_obj[name] = sub_cls
            to_check.append(sub_cls)
