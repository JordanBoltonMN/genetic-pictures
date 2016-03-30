import random

import numpy as np
from PIL import Image

import evaluators
import species
from mp import MPWorldRepresentation, MPWorldFitness


class BaseWorld(object):
    _representation = None

    def __init__(self,
                 problem,
                 species_name,
                 size=None,
                 inhabitants=None,
                 **kwargs):
        self.problem = problem

        if species_name not in species.name_to_obj:
            error = "Unknown species '{0}'".format(species_name)
            raise ValueError(error)
        else:
            self.species_cls = species.name_to_obj[species_name]

        if inhabitants is None:
            if size is None:
                raise ValueError("size must be given if inhabitants isn't")
            self.inhabitants = self.generate_inhabitants(size)
        else:
            self.inhabitants = self.load_inhabitants(inhabitants)
            self.update_fitness()

    @property
    def representation(self):
        if self._representation is None:
            self._representation = self.create_representation()
        return self._representation

    @representation.setter
    def representation(self, value):
        self._representation = None

    def generate_inhabitants(self, size):
        return [self.species_cls(self.problem) for _ in range(size)]

    def load_inhabitants(self, inhabitants):
        return [self.species_cls(self.problem, **d) for d in inhabitants]

    def setup(self):
        print "\tCalculating fitness for current world.",
        self.update_fitness()
        print "Done."

    def create_representation(self):
        image = self.problem.image
        (height, width, _) = image.shape
        image_dimensions = (height, width)

        base = np.full((height, width, 4), 0, dtype=image.dtype)
        result = Image.fromarray(base)

        pool = MPWorldRepresentation(image_dimensions)
        representation = pool(self.inhabitants)
        self.representation = representation
        return representation

    def update_fitness(self):
        target = self.problem.image
        representation = self.representation.convert("RGB")
        np_representation = np.array(representation)

        pool = MPWorldFitness()
        self.fitness = pool((target, np_representation))

    def next_generation(self):
        # the private method ensures it's both implemented and that
        # that representation cache is cleared on call
        self.representation = None
        self._next_generation()

    def _next_generation(self):
        raise NotImplementedError()

    def json(self):
        inhabitants = [individual.json() for individual in self.inhabitants]

        return {
            "world_name" : self.__class__.__name__,
            "species_name" : self.species_cls.__name__,
            "inhabitants" : inhabitants,
        }

class IterativeWorld(BaseWorld):
    # def _next_generation(self):
    #     print "\tMutating individuals.",
    #     mutated_json = self._mutate_inhabitants()
    #     print "Done."

    #     print "\tCreating new world.",
    #     potential_world = self._potential_world(mutated_json)
    #     print "Done."

    #     if self._representation is None:
    #         print "\tUpdating representation for current world.",
    #         self.create_representation()
    #         print "Done."


    #     print "\tCalculating fitness for new world.",
    #     potential_world.update_fitness()
    #     print "Done."

    #     print "Is {current} > {new}? {result}".format(
    #         current=self.fitness,
    #         new=potential_world.fitness,
    #         result=self.fitness > potential_world.fitness,
    #     )
    #     if self.fitness > potential_world.fitness:
    #         print "\tUpdating world."
    #         self.inhabitants = potential_world.inhabitants
    #         self.fitness = potential_world.fitness
    #         print "Done."
    #     else:
    #         print "\tWorld remains unchanged."

    def _next_generation(self):
        inhabitants = [individual.json() for individual in self.inhabitants]
        mutant_index = random.randint(0, len(inhabitants) - 1)
        inhabitants[mutant_index] = self.inhabitants[mutant_index].mutate()

        test_world = self.__class__(self.problem, inhabitants=inhabitants)
        test_world.update_fitness()

        print self.fitness, test_world.fitness, self.fitness > test_world.fitness

        # a lower fitness means less different
        if self.fitness > test_world.fitness:
            self.inhabitants = test_world.inhabitants
            self.fitness = test_world.fitness
            print "better"
        else:
            print "worse"

        # print "\tCreating new world.",
        # potential_world = self._potential_world(mutated_json)
        # print "Done."

        # if self._representation is None:
        #     print "\tUpdating representation for current world.",
        #     self.create_representation()
        #     print "Done."


        # print "\tCalculating fitness for new world.",
        # potential_world.update_fitness()
        # print "Done."

        # print "Is {current} > {new}? {result}".format(
        #     current=self.fitness,
        #     new=potential_world.fitness,
        #     result=self.fitness > potential_world.fitness,
        # )
        # if self.fitness > potential_world.fitness:
        #     print "\tUpdating world."
        #     self.inhabitants = potential_world.inhabitants
        #     self.fitness = potential_world.fitness
        #     print "Done."
        # else:
        #     print "\tWorld remains unchanged."

    def _mutate_inhabitants(self):
        return [i.mutate() for i in self.inhabitants]

    def _potential_world(self, inhabitants_json):
        return self.__class__(self.problem, inhabitants=inhabitants_json)


class TriangleWorld(IterativeWorld):
    def __init__(self, problem, **kwargs):
        kwargs.setdefault("species_name", "Triangle")
        kwargs.setdefault("evaluator_name", "RGBDifference")
        kwargs.setdefault("size", 128)
        super(TriangleWorld, self).__init__(problem, **kwargs)


name_to_obj = {}
to_check = [BaseWorld]
while to_check:
    cls = to_check.pop()
    for sub_cls in cls.__subclasses__():
        name = sub_cls.__name__
        if name not in name_to_obj:
            name_to_obj[name] = sub_cls
            to_check.append(sub_cls)
