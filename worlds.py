import random

import evaluators
import populations


class BaseWorld(object):
    def __init__(self,
                 problem,
                 population_name,
                 evaluator_name,
                 mutation_rate=0.1,
                 size=None,
                 inhabitants=None,
                 **kwargs):
        self.problem = problem
        self.mutation_rate = mutation_rate

        if population_name not in populations.name_to_obj:
            error = "Unknown population_name '{0}'".format(population_name)
            raise ValueError(error)
        else:
            self.population_cls = populations.name_to_obj[population_name]

        if evaluator_name not in evaluators.name_to_obj:
            error = "Unknown evaluator '{0}'".format(evaluator_name)
            raise ValueError(error)
        else:
            evaluator_kwargs = kwargs.get("evaluator", {})
            evaluator_cls = evaluators.name_to_obj[evaluator_name]
            self.evaluator = evaluator_cls(**evaluator_kwargs)

        if inhabitants is None:
            if size is None:
                raise ValueError("size must be given if inhabitants isn't")
            self.inhabitants = self.generate_inhabitants(size)
        else:
            self.inhabitants = self.load_inhabitants(inhabitants)

    def generate_inhabitants(self, size):
        return [self.population_cls(self.problem) for _ in range(size)]

    def load_inhabitants(self, inhabitants):
        return [self.population_cls(self.problem, **d) for d in inhabitants]

    def setup(self):
        self.inhabitants = self.update_fitness(self.inhabitants)

    def update_fitness(self, inhabitants):
        return self.evaluator(self.problem.image, inhabitants)

    def breed(self):
        raise NotImplementedError()

    def json(self):
        inhabitants = [population.json() for population in self.inhabitants]

        return {
            "world_name" : self.__class__.__name__,
            "elite_count" : self.elite_count,
            "tournament_size" : self.tournament_size,
            "mutation_rate" : self.mutation_rate,

            "population_name" : self.population_cls.__name__,
            "inhabitants" : inhabitants,

            "evaluator_name" : self.evaluator.__class__.__name__,
            "evaluator" : self.evaluator.json(),
        }

class TournamentWorld(BaseWorld):
    def __init__(self, problem, **kwargs):
        kwargs.setdefault("population_name", "EllipsePopulation")
        kwargs.setdefault("evaluator_name", "RGBDifference")
        kwargs.setdefault("size", 100)
        super(TournamentWorld, self).__init__(problem, **kwargs)

        size = kwargs["size"]
        kwargs.setdefault("tournament_size", int(size * 0.1) or 1)
        kwargs.setdefault("elite_count", int(size * 0.01) or 1)

        self.tournament_size = kwargs["tournament_size"]
        self.elite_count = kwargs["elite_count"]

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

    def json(self):
        d = super(TournamentWorld, self).json()
        d.update(
            elite_count=self.elite_count,
            tournament_size=self.tournament_size,
        )
        return d

name_to_obj = {}
to_check = [BaseWorld]
while to_check:
    cls = to_check.pop()
    for sub_cls in cls.__subclasses__():
        name = sub_cls.__name__
        if name not in name_to_obj:
            name_to_obj[name] = sub_cls
            to_check.append(sub_cls)
