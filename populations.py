import random
import numpy as np
from multiprocessing import Pool

import species


class BasePopulation(object):
    def __init__(self, problem, **kwargs):
        init_kwargs = self.default_kwargs()
        init_kwargs.update(kwargs)
        self.problem = problem

        self.size = init_kwargs["size"]
        self.tournament_size = init_kwargs["tournament_size"]
        self.elite_count = init_kwargs["elite_count"]
        self.mutation_rate = init_kwargs["mutation_rate"]
        self.processes = init_kwargs["processes"]

        self.species_name = init_kwargs["species_name"]
        if self.species_name not in species.name_to_obj:
            error = "Unknown species_name '{0}'".format(self.species_name)
            raise ValueError(error)
        else:
            self.species_cls = species.name_to_obj[self.species_name]

        if "pool" not in init_kwargs:
            self.pool = self.generate_pool()
        else:
            self.pool = self.load_pool(init_kwargs["pool"])

    def default_kwargs(self):
        return {
            "species_name" : "Ellipse",
            "size" : 1000,
            "tournament_size" : 100,
            "elite_count" : 10,
            "mutation_rate" : 0.05,
            "processes" : 4,
        }

    def individuals(self):
        return self.pool

    def generate_pool(self):
        return [self.species_cls(self.problem) for _ in range(self.size)]

    def load_pool(self, pool):
        return [self.species_cls(self.problem, **d) for d in pool]

    def update_fitness(self):
        individuals = [(self.problem, i) for i in self.pool]
        thread_pool = Pool(processes=4)
        self.pool = thread_pool.map(update_individual_fitness, individuals)
        self.pool.sort(reverse=True, key=lambda individual: individual.fitness)

    def breed(self):
        # sort by highest fitness to lowest
        self.update_fitness()
        self.pool.sort(reverse=True, key=lambda individual: individual.fitness)

        # grab first 'elite_count' individuals
        n_elites = self.elite_count
        new_pool = self.pool[:n_elites]

        while len(new_pool) != len(self.pool):
            i1 = self.tournament()
            i2 = self.tournament()
            while i1 is i2:
                i2 = self.tournament()

            child = i1.breed_with(i2)
            if self.should_mutate():
                child.mutate()

            new_pool.append(child)

        self.pool = new_pool

    def tournament(self):
        size = self.tournament_size
        tourny = [random.choice(self.pool) for _ in range(size)]
        tourny.sort(key=lambda individual: individual.fitness, reverse=True)
        return tourny[0]

    def should_mutate(self):
        return random.random() <= self.mutation_rate

    def json(self):
        return {
            "elite_count" : self.elite_count,
            "tournament_size" : self.tournament_size,
            "mutation_rate" : self.mutation_rate,
            "processes" : self.processes,

            "species_name" : self.species_name,
            "pool" : [individual.json() for individual in self.pool]
        }


def update_individual_fitness((problem, individual)):
    """
    used by multiprocessing.Pool.map, which can only take a single value,
    but the function needs two arguments which is why the syntax is weird
    """
    mask = individual.create_mask(problem)
    overlap = np.bitwise_and(problem.image, mask)
    individual.fitness = problem.evaluator(problem.image, overlap)
    return individual


name_to_obj = {BasePopulation.__name__ : BasePopulation}
for cls in BasePopulation.__subclasses__():
    name_to_obj[cls.__name__] = cls
