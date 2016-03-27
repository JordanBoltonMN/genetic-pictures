import random

import evaluators
import populations

class BaseWorld(object):
    def __init__(self, problem,
        population_name="EllipsePopulation", evaluator_name="RGBDifference",
        size=100, mutation_rate=0.1, tournament_size=None, elite_count=None,
        inhabitants=None,
        **kwargs):
        if tournament_size is None:
            tournament_size = int(size * 0.1) or 1
        if elite_count is None:
            elite_count = int(size * 0.01) or 1

        self.problem = problem

        if population_name not in populations.name_to_obj:
            error = "Unknown population_name '{0}'".format(population_name)
            raise ValueError(error)
        else:
            self.population_name = population_name
            self.populations_cls = populations.name_to_obj[population_name]

        if inhabitants is None:
            self.size = size
            self.inhabitants = self.generate_inhabitants()
        else:
            self.size = len(inhabitants)
            self.inhabitants = self.load_inhabitants(inhabitants)

        assert self.size >= 2

        self.tournament_size = tournament_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate

        self.evaluator_name = evaluator_name
        self.evaluator = self.create_evaluator(**kwargs)

    def setup(self):
        self.inhabitants = self.update_fitness(self.inhabitants)

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
        return self.inhabitants

    def generate_inhabitants(self):
        return [self.populations_cls(self.problem) for _ in range(self.size)]

    def load_inhabitants(self, inhabitants):
        return [self.populations_cls(self.problem, **d) for d in inhabitants]

    def update_fitness(self, inhabitants):
        return self.evaluator(self.problem.image, inhabitants)

    def breed(self):
        # sort by fitness for elite_count
        self.evaluator.sort(self.inhabitants)

        # grab first 'elite_count' individuals
        n_elites = self.elite_count

        new_inhabitants = self.inhabitants[:n_elites]

        while len(new_inhabitants) != len(self.inhabitants):
            i1 = self.tournament()
            i2 = self.tournament()
            while i1 is i2:
                i2 = self.tournament()

            child = i1.breed_with(i2)
            child.mutate(self.mutation_rate)
            new_inhabitants.append(child)

        self.inhabitants = self.update_fitness(new_inhabitants)

    def tournament(self):
        size = self.tournament_size
        tourney = [random.choice(self.inhabitants) for _ in range(size)]
        self.evaluator.sort(tourney)
        return tourney[0]

    def should_mutate(self):
        return random.random() <= self.mutation_rate

    def json(self):
        inhabitants = [population.json() for population in self.inhabitants]

        return {
            "world_name" : self.__class__.__name__,
            "elite_count" : self.elite_count,
            "tournament_size" : self.tournament_size,
            "mutation_rate" : self.mutation_rate,

            "population_name" : self.population_name,
            "inhabitants" : inhabitants,

            "evaluator_name" : self.evaluator_name,
            "evaluator" : self.evaluator.json(),
        }

name_to_obj = {"BaseWorld" : BaseWorld}
to_check = [BaseWorld]
while to_check:
    cls = to_check.pop()
    for sub_cls in cls.__subclasses__():
        name = sub_cls.__name__
        if name not in name_to_obj:
            name_to_obj[name] = sub_cls
            to_check.append(sub_cls)
