import random

import numpy as np

import species


class BasePopulation(object):
    def __init__(self, problem, species_cls, individuals=None, size=100,
                 **kwargs):
        self.problem = problem
        self.species_cls = species_cls

        if individuals is None:
            self.individuals = self.create_individuals(size)
        else:
            self.individuals = self.load_individuals(individuals)

    def __iter__(self):
        return iter(self.individuals)

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, index):
        return self.individuals[index]

    def __setitem__(self, index, value):
        self.individuals[index] = value

    @property
    def fitness(self):
        return sum([individual.fitness for individual in self])

    def create_individuals(self, size):
        individuals = []
        for i in range(size):
            individuals.append(self.species_cls(self.problem))
        return individuals

    def load_individuals(self, individuals):
        return [self.species_cls(self.problem, **d) for d in individuals]

    def new_population(self, individuals):
        return self.__class__(self.problem, individuals=individuals)

    def breed_with(self, other):
        assert len(self) == len(other)

        children = []
        pool1 = list(self)
        pool2 = list(other)
        random.shuffle(pool1)
        random.shuffle(pool2)

        for i, individual1 in enumerate(pool1):
            individual2 = pool2[i]
            children.append(individual1.breed_with(individual2))

        json_children = [child.json() for child in children]
        return self.new_population(json_children)

    def mutate(self, mutation_rate):
        for individual in self:
            if random.random() < mutation_rate:
                individual.mutate()

    def create_representation(self):
        mask = np.zeros_like(self.problem.image)
        for individual in self:
            representation = individual.create_representation()
            np.copyto(mask, representation, where=representation>0)

        return mask

    def json(self):
        return {
            "individuals" : [individual.json() for individual in self],
        }


class EllipsePopulation(BasePopulation):
    def __init__(self, problem, **kwargs):
        super(EllipsePopulation, self).__init__(
            problem,
            species.Ellipse,
            **kwargs
        )


name_to_obj = {}
to_check = [BasePopulation]
while to_check:
    cls = to_check.pop()
    for sub_cls in cls.__subclasses__():
        name = sub_cls.__name__
        if name not in name_to_obj:
            name_to_obj[name] = sub_cls
            to_check.append(sub_cls)
