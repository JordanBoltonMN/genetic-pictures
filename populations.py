from __future__ import division
from math import ceil, floor
import numpy as np
import random

import mp
from species import TrianglePool


class Population(object):
    def __init__(self, problem, size=128,
                 mutation_rate=0.01, selection_rate=0.15):
        self.problem = problem
        self.size = size
        self.mutation_rate = mutation_rate
        self.selection_rate = selection_rate

        self.inhabitants = self.create_inhabitants(size)

    def setup(self):
        self.update_fitness()

    def update_fitness(self):
        pool = mp.MPIndividualFitness(self.problem.image_np)
        pool(self.inhabitants)
        self.inhabitants.sort(key=lambda individual: individual.fitness)

    def create_inhabitants(self, size):
        dimensions = self.problem.image_dimensions
        return [TrianglePool(dimensions) for _ in range(size)]

    def generation(self):
        tournament_size = self.size // 10
        children = []

        while len(children) != len(self.inhabitants):
            pa = self.tournament_selection(tournament_size)
            ma = self.tournament_selection(tournament_size)
            while pa is ma:
                ma = self.tournament_selection(tournament_size)

            child = pa.mate(ma)
            children.append(child)

        self.inhabitants = children
        self.update_fitness()

    def tournament_selection(self, tournament_size):
        contestants = []
        for _ in range(tournament_size):
            contestants.append(random.choice(self.inhabitants))

        contestants.sort(key=lambda individual: individual.fitness)
        return contestants[0]

    def representation(self):
        return self.inhabitants[0].representation().astype(np.uint8)

    def json(self):
        return {
            "inhabitants" : [i.json() for i in self.inhabitants],
            "mutation_rate" : self.mutation_rate
        }
