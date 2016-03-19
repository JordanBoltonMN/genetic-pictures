import glob
import hashlib
import json
import os

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
            print "Setting up problem.",
            self.setup_problem()
            print "Done."

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
            "image_name" : image_name,
            "image" : cv2.imread(image_name),
            "image_md5" : md5(image_name),
        }

    def kwargs_from_snapshot_name(self, snapshot_name):
        # was given a specific snapshot
        if "_gen_" in snapshot_name and os.path.isfile(snapshot_name):
            snapshot_fname = snapshot_name
        else:
            # was given a generic name, find the latest generation snapshot
            similar = glob.glob("{0}_gen_*.json".format(snapshot_name))
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
        population_name = kwargs["population_name"]
        if population_name not in populations.name_to_obj:
            error = "Unknown population '{0}'".format(population_name)
            raise ValueError(error)
        else:
            return populations.name_to_obj[population_name](self, **kwargs)

    def save_snapshot(self):
        fname = self.generation_filename() + ".json"

        print "Creating snapshot {0}.".format(fname)
        print "\tGenerating json.",
        # used on a Problem's init
        d = {
            "problem" : self.json(),
            "population" : self.population.json()
        }
        print "Done."

        print "\tSaving json.",
        with open(fname, "w") as f:
            json.dump(d, f, sort_keys=True, indent=4)
        print "Done."

    def should_save_image(self):
        if not self.picture_every:
            return False
        else:
            return (self.current_generation % self.picture_every) == 0

    def save_image(self):
        fname = self.generation_filename() + ".png"
        result = np.zeros_like(self.image)

        print "Creating image {0}:".format(fname)
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
        result.save(fname)

        print "Done."

    def generation_filename(self):
        root = os.path.basename(self.dst_name)
        padded_generation = str(self.current_generation).zfill(7)

        return "{root}_gen_{padded_generation}".format(
            root=root,
            padded_generation=padded_generation,
        )

    def json(self):
        return {
            "image_name" : self.image_name,
            "dst_name" : self.dst_name,
            "image_md5" : self.image_md5,
            "current_generation" : self.current_generation,
            "generations" : self.generations,
            "picture_every" : self.picture_every,
            "population_name" : self.population.__class__.__name__,
        }

def create_images(glob_pattern=None, problem_cls=None, limited_range=None):
    if glob_pattern is None:
        glob_pattern = "*_gen_*"

    if problem_cls is None:
        problem_cls = Problem

    for filepath in glob.glob(glob_pattern):
        problem = problem_cls(snapshot_name=filepath)
        if limited_range and problem.current_generation not in limited_range:
            continue
        problem.save_image()

if __name__ == "__main__":
    problem = Problem(
        image_name="target.png",
        dst_name="target",
        generations=20,
    )
    problem = Problem(
        snapshot_name="target.png",
        dst_name="target",
        generations=20,
    )
    problem.run()
