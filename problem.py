import calendar
import glob
import hashlib
import json
import os
import random
import time

import numpy as np
from PIL import Image

from populations import Population

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class Problem(object):
    def __init__(self, image_name, seed=None):
        self.image = Image.open(image_name).convert("RGBA")
        self.image_np = np.array(self.image, dtype=np.uint8)
        self.dst_name = image_name

        (height, width, _) = self.image_np.shape
        self.image_dimensions = (height, width)

        self.current_generation = 0
        self.generations = 10000
        self.seed = seed or calendar.timegm(time.gmtime())

        # look in the method self.setup
        self.population = None

    def run(self):
        self.setup()
        self.save_snapshot()
        self.save_image()
        self.current_generation = 1

        while self.current_generation <= self.generations:
            print self.current_generation
            self.population.generation()

            if self.should_save_snapshot():
                self.save_snapshot()
            if self.should_save_image():
                self.save_image()

            self.current_generation += 1

    def setup(self):
        self.population = Population(self)
        self.population.setup()

    def should_save_snapshot(self):
        return (self.current_generation % 100) == 0

    def should_save_image(self):
        return (self.current_generation % 100) == 0

    def save_snapshot(self):
        filepath = self._generate_filepath("json")
        with open(filepath, "w") as f:
            json.dump(self.json(), f, sort_keys=True)

    def save_image(self):
        filepath = self._generate_filepath("png")
        image_np = self.population.representation()
        image = Image.fromarray(image_np, "RGBA")
        image.save(filepath)

    def json(self):
        return {
            "population" : self.population.json(),
            "generations" : self.generations,
            "current_generation" : self.current_generation,
            "seed" : self.seed,
        }

    def _results_directory(self):
        directory = os.path.join(os.path.dirname(__file__), "results")
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def _generate_filepath(self, extension):
        root = os.path.basename(self.dst_name)
        padded_generation = str(self.current_generation).zfill(7)
        file_name = "{root}_gen_{padded_generation}.{extension}".format(
            root=root,
            padded_generation=padded_generation,
            extension=extension,
        )

        return os.path.join(self._results_directory(), file_name)

if __name__ == "__main__":
    problem = Problem("obama.png")
    problem.run()
