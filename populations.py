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
        stud_count = int(floor(len(self.inhabitants) * self.selection_rate))
        random_count = int(ceil(1 / self.selection_rate))

        children = []
        for stud in self.inhabitants[:stud_count]:
            for _ in range(random_count):
                mate_index = random.randint(0, stud_count - 1)
                mate = self.inhabitants[mate_index]

                while stud is mate:
                    mate_index = random.randint(0, stud_count - 1)
                    mate = self.inhabitants[mate_index]

                children.append(stud.breed(mate))

        children = children[:len(self.inhabitants)]

        while len(children) != len(self.inhabitants):
            stud = random.choice(self.inhabitants[:stud_count])
            mate_index = random.randint(0, stud_count - 1)
            mate = self.inhabitants[mate_index]

            while stud is mate:
                mate_index = random.randint(0, stud_count - 1)
                mate = self.inhabitants[mate_index]

            children.append(stud.breed(mate))

        for child in children:
            child.mutate(self.mutation_rate)

        self.inhabitants = children
        self.update_fitness()

    def representation(self):
        return self.inhabitants[0].representation().astype(np.uint8)

    def json(self):
        return {
            "inhabitants" : [i.json() for i in self.inhabitants],
            "mutation_rate" : self.mutation_rate
        }
