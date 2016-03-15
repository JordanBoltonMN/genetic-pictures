import numpy as np

class BaseEvaluator(object):
    def __call__(self, image, overlap):
        return self.individual_fitness(image, overlap)

    def individual_fitness(self, image, overlap):
        fitness = 0
        for (y, x) in np.argwhere(overlap.any(-1)):
            rgb1 = image[y][x]
            rgb2 = overlap[y][x]
            fitness += self.pixel_fitness(rgb1, rgb2)
        return fitness

    def pixel_fitness(self, rgb1, rgb2):
        """ euclidian distance where (x,y,z) is (r,g,b) """
        return sum(map(lambda (a, b): abs(a - b), zip(rgb1, rgb2)))

name_to_obj = {BaseEvaluator.__name__ : BaseEvaluator}
for cls in BaseEvaluator.__subclasses__():
    name_to_obj[cls.__name__] = cls
