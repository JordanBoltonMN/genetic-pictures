import random

import evaluators
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
            self.size = len(self.pool)

        self.evaluator_name = init_kwargs["evaluator_name"]
        self.evaluator = self.create_evaluator(**init_kwargs)

    def default_kwargs(self):
        size = 1000
        tournament_size = int(size * 0.1)
        elite_count = int(size * 0.001)

        return {
            "species_name" : "Ellipse",
            "evaluator_name" : "WeightedRGBDifference",

            "size" : size,
            "tournament_size" : tournament_size,
            "elite_count" : elite_count,
            "mutation_rate" : 0.1,
        }

    def create_evaluator(self, **kwargs):
        # instantiate the evaluator for the given name
        # look in evaluators.py to see how name_to_obj is generated
        evaluator_name = kwargs["evaluator_name"]
        evaluator_kwargs = kwargs.get("evaluator", {})

        if evaluator_name not in evaluators.name_to_obj:
            error = "Unknown evaluator '{0}'".format(evaluator_name)
            raise ValueError(error)
        else:
            evaluator_cls = evaluators.name_to_obj[evaluator_name]
            return evaluator_cls(**evaluator_kwargs)

    def individuals(self):
        return self.pool

    def generate_pool(self):
        return [self.species_cls(self.problem) for _ in range(self.size)]

    def load_pool(self, pool):
        return [self.species_cls(self.problem, **d) for d in pool]

    def update_fitness(self):
        self.pool = self.evaluator(self.problem.image, self.pool)
        self._sort_pool(self.pool)

    def breed(self):
        # sort by fitness for elite_count
        self._sort_pool(self.pool)

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
        self.update_fitness()

    def tournament(self):
        size = self.tournament_size
        tourney = [random.choice(self.pool) for _ in range(size)]
        self._sort_pool(tourney)
        return tourney[0]

    def should_mutate(self):
        return random.random() <= self.mutation_rate

    def json(self):
        return {
            "species_name" : self.species_name,
            "evaluator_name" : self.evaluator_name,

            "elite_count" : self.elite_count,
            "tournament_size" : self.tournament_size,
            "mutation_rate" : self.mutation_rate,

            "pool" : [individual.json() for individual in self.pool],

            "evaluator" : self.evaluator.json(),
        }

    def _sort_pool(self, pool):
        pool.sort(
            key=lambda individual: individual.fitness,
            reverse=self.evaluator.reverse_sort,
        )

name_to_obj = {BasePopulation.__name__ : BasePopulation}
for cls in BasePopulation.__subclasses__():
    name_to_obj[cls.__name__] = cls
