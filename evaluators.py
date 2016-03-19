import numpy as np

class BaseEvaluator(object):
    def __init__(self, reverse_sort=False, **kwargs):
        self.reverse_sort = reverse_sort

    def __call__(self, image, representation):
        raise NotImplementedError()

class RGBDifference(BaseEvaluator):
    def __call__(self, image, representation):
        return self.individual_fitness(image, representation)

    def individual_fitness(self, image, representation):
        fitness = 0
        overlap = np.bitwise_and(image, representation)
        for (y, x) in np.argwhere(overlap.any(-1)):
            rgb1 = image[y][x]
            rgb2 = representation[y][x]
            fitness += self.pixel_fitness(rgb1, rgb2)
        return fitness

    def pixel_fitness(self, rgb1, rgb2):
        """ sum the difference between color octals """
        return sum(map(lambda (a, b): abs(int(a) - int(b)), zip(rgb1, rgb2)))

name_to_obj = {}
for cls in BaseEvaluator.__subclasses__():
    name_to_obj[cls.__name__] = cls
