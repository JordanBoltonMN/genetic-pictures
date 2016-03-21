import calendar
import glob
import hashlib
import json
import os
import random
import time

import cv2
import numpy as np
from PIL import Image

import populations

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class Problem(object):
    def __init__(self, image_name=None, snapshot_name=None, **kwargs):
        if not image_name and not snapshot_name:
            raise ValueError("requires either image_name or snapshot_name")

        init_kwargs = self.default_kwargs()
        if image_name:
            init_kwargs.update(self.kwargs_from_image_name(image_name))
        elif snapshot_name:
            init_kwargs.update(self.kwargs_from_snapshot_name(snapshot_name))
        else:
            raise Exception("never should reach here")

        init_kwargs.update(kwargs)

        self.seed = init_kwargs["seed"]
        random.seed(self.seed)

        self.current_generation = init_kwargs["current_generation"]
        self.generations = init_kwargs["generations"]
        self.picture_every = init_kwargs["picture_every"]
        self.image_name = init_kwargs["image_name"]
        self.image = init_kwargs["image"]
        self.image_md5 = init_kwargs["image_md5"]
        self.dst_name = init_kwargs.get("dst_name", self.image_name)
        self.height, self.width, _ = self.image.shape

        self.population = self.create_population(**init_kwargs)

    def run(self):
        if self.current_generation == 0:
            print "Performing setup for generation 0."
            self.setup_problem()

        for _ in range(self.current_generation, self.generations):
            self.next_generation()

    def setup_problem(self):
        self.population.update_fitness()
        self.save_snapshot()
        if self.should_save_image():
            self.save_image()

    def next_generation(self):
        self.current_generation += 1
        print "Starting generation: {0}".format(self.current_generation)
        print "\tBreeding.",
        self.population.breed()
        print "Done"

        self.save_snapshot()

        if self.should_save_image():
            self.save_image()

    def default_kwargs(self):
        return {
            "current_generation" : 0,
            "generations" : 10,
            "picture_every" : 1,
            "population_name" : "BasePopulation",
        }

    def kwargs_from_image_name(self, image_name):
        return {
            "seed" : calendar.timegm(time.gmtime()),
            "image_name" : image_name,
            "image" : cv2.imread(image_name),
            "image_md5" : md5(image_name),
        }

    def kwargs_from_snapshot_name(self, snapshot_name):
        # was given a specific snapshot
        results_directory = self.results_directory()

        possible_filepath = os.path.join(results_directory, snapshot_name)
        if os.path.isfile(possible_filepath):
            snapshot_fname = possible_filepath
        else:
            # was given a generic name, find the latest generation snapshot
            snapshot_pattern = "{0}_gen_*.json".format(snapshot_name)
            glob_pattern = os.path.join(
                results_directory,
                snapshot_pattern,
            )
            similar = glob.glob(glob_pattern)
            # no snapshots exist for the given generic name
            if not similar:
                error = "No snapshots of name {0} exist".format(snapshot_name)
                raise ValueError(error)

            # snapshot of the latest generation
            snapshot_fname = max(similar)

        with open(snapshot_fname, "r") as f:
            snapshot = json.load(f)

        image_name = snapshot["problem"]["image_name"]
        result = self.kwargs_from_image_name(image_name)
        result.update(snapshot["problem"])
        result.update(snapshot["population"])

        # double check that the loaded snapshot is for the right image
        new_md5 = md5(image_name)
        old_md5 = result["image_md5"]
        if new_md5 != old_md5:
            error = "md5 mismatch for file {0} and the snapshot {1}".format(
                image_name,
                snapshot_name
            )
            raise ValueError(error)

        return result

    def create_population(self, **kwargs):
        # instantiate the population for the given name
        # look in populations.py to see how name_to_obj is generated
        population_name = kwargs.get("population_name")

        if population_name not in populations.name_to_obj:
            error = "Unknown population '{0}'".format(population_name)
            raise ValueError(error)
        else:
            return populations.name_to_obj[population_name](self, **kwargs)

    def save_snapshot(self):
        filepath = self.generate_filepath("json")

        print "Creating snapshot {0}.".format(filepath)
        print "\tGenerating json.",
        # used on a Problem's init
        d = {
            "problem" : self.json(),
            "population" : self.population.json()
        }
        print "Done."

        print "\tSaving json.",
        with open(filepath, "w") as f:
            json.dump(d, f, sort_keys=True, indent=4)
        print "Done."

    def should_save_image(self):
        if not self.picture_every:
            return False
        else:
            return (self.current_generation % self.picture_every) == 0

    def save_image(self):
        filepath = self.generate_filepath("png")
        result = np.zeros_like(self.image)

        print "Creating image {0}:".format(filepath)
        print "\tGenerating Image.",

        for individual in self.population.individuals():
            # 3D numpy array, (y, x, RGB)
            representation = individual.create_representation()
            # blit pixels to screen
            for (y, x) in np.argwhere(representation.any(axis=-1)):
                result[y][x] = representation[y][x]

        print "Done"
        print "\tSaving Image",

        result = Image.fromarray(result, mode="RGB")
        result.save(filepath)

        print "Done."

    def generate_filepath(self, extension):
        results_directory = self.results_directory()

        root = os.path.basename(self.dst_name)
        padded_generation = str(self.current_generation).zfill(7)
        file_name = "{root}_gen_{padded_generation}.{extension}".format(
            root=root,
            padded_generation=padded_generation,
            extension=extension,
        )

        return os.path.join(results_directory, file_name)

    def results_directory(self):
        directory = os.path.join(os.path.dirname(__file__), "results")
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def snapshots_glob(self, pattern=None):
        if pattern is None:
            pattern = "*_gen_*"

        return glob.iglob(os.path.join(self.results_directory(), pattern))

    def json(self):
        return {
            "seed" : self.seed,
            "image_name" : self.image_name,
            "dst_name" : self.dst_name,
            "image_md5" : self.image_md5,
            "current_generation" : self.current_generation,
            "generations" : self.generations,
            "picture_every" : self.picture_every,
            "population_name" : self.population.__class__.__name__,
        }


def make_missing_images(problem):
    def strip_gens(paths):
        result = set()

        for p in paths:
            start = p.find("_gen_") + len("_gen_")
            stop = p.rfind(".")
            result.add(p[start:stop])

        return result

    json_pattern = problem.dst_name + "_gen_*.json"
    json_gens = strip_gens(problem.snapshots_glob(json_pattern))

    picture_pattern = problem.dst_name + "_gen_*.png"
    picture_gens = strip_gens(problem.snapshots_glob(picture_pattern))

    missing_picture_gens = json_gens - picture_gens
    for gen in missing_picture_gens:
        snapshot_name = "{0}_gen_{1}.json".format(
            problem.dst_name,
            gen
        )
        new_problem = Problem(snapshot_name=snapshot_name)
        new_problem.save_image()


if __name__ == "__main__":
    problem = Problem(
        image_name="circles.png",
        dst_name="rgb",
        generations=10,
        size=10000,
        evaluator_name="RGBDifference",
    )
    problem.run()

    problem = Problem(
        image_name="circles.png",
        dst_name="clilab",
        generations=10,
        size=10000,
        evaluator_name="CIELabDifference",
    )
    problem.run()

    problem = Problem(
        image_name="circles.png",
        dst_name="rgb_weighted",
        generations=10,
        size=10000,
        evaluator_name="WeightedRGBDifference",
    )
    problem.run()

    problem = Problem(
        image_name="circles.png",
        dst_name="clilab_weighted",
        generations=10,
        size=10000,
        evaluator_name="WeightedCIELabDifference",
    )
    problem.run()
