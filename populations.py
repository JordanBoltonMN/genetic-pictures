import random

import evaluators
import species

class BasePopulation(object):
    def __init__(self, problem,
        species_name="Ellipse", evaluator_name="RGBDifference", size=1000,
        mutation_rate=0.1, tournament_size=None, elite_count=None, pool=None,
        **kwargs):
        if tournament_size is None:
            tournament_size = int(size * 0.1)
        if elite_count is None:
            elite_count = int(size * 0.01)

        self.problem = problem

        if species_name not in species.name_to_obj:
            error = "Unknown species_name '{0}'".format(species_name)
            raise ValueError(error)
        else:
            self.species_name = species_name
            self.species_cls = species.name_to_obj[species_name]

        if pool is None:
            self.size = size
            self.pool = self.generate_pool()
        else:
            self.size = len(pool)
            self.pool = self.load_pool(pool)

        self.tournament_size = tournament_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate

        self.evaluator_name = evaluator_name
        self.evaluator = self.create_evaluator(**kwargs)

    def create_evaluator(self, evaluator_name=None, **kwargs):
        # instantiate the evaluator for the given name
        # look in evaluators.py to see how name_to_obj is generated
        if evaluator_name is None:
            evaluator_name = self.evaluator_name

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
            "elite_count" : self.elite_count,
            "tournament_size" : self.tournament_size,
            "mutation_rate" : self.mutation_rate,

            "species_name" : self.species_name,
            "pool" : [individual.json() for individual in self.pool],

            "evaluator_name" : self.evaluator_name,
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
