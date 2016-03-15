import glob
import hashlib
import json
import os

import cv2
import numpy as np
from PIL import Image

import evaluators
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
        self.height, self.width, _ = self.image.shape

        self.evaluator = self.create_evaluator(**init_kwargs)
        self.population = self.create_population(**init_kwargs)

    def run(self):
        for _ in range(self.current_generation, self.generations):
            self.next_generation()
            self.save_snapshot()

    def next_generation(self):
        print "Starting generation: {0}".format(self.current_generation)
        print "\tBreeding.",
        self.population.breed()
        print "Done"
        self.current_generation += 1

        if (self.current_generation % self.picture_every) == 0:
            self.save_image()

    def default_kwargs(self):
        return {
            "current_generation" : 0,
            "generations" : 2,
            "picture_every" : 1,
            "evaluator_name" : "BaseEvaluator",
            "population_name" : "BasePopulation",
        }

    def kwargs_from_image_name(self, image_name):
        return {
            "image_name" : image_name,
            "image" : cv2.imread(image_name),
            "image_md5" : md5(image_name),
        }

    def kwargs_from_snapshot_name(self, snapshot_name):
        # all which match the generic name
        similar = glob.glob("{0}_gen_*.json".format(snapshot_name))
        if not similar:
            error = "No snapshots of name {0} exist".format(snapshot_name)
            raise ValueError(error)

        # highest generation file
        snapshot_fname = max(similar)
        with open(snapshot_fname, "r") as f:
            snapshot = json.load(f)

        image_name = snapshot["problem"]["image_name"]
        result = self.kwargs_from_image_name(image_name)
        result.update(snapshot["population"])

        new_md5 = md5(image_name)
        old_md5 = result["image_md5"]
        if new_md5 != old_md5:
            error = "md5 mismatch for file {0} and the snapshot {1}".format(
                image_name,
                snapshot_name
            )
            raise ValueError(error)

        return result

    def create_evaluator(self, **kwargs):
        evaluator_name = kwargs["evaluator_name"]
        if evaluator_name not in evaluators.name_to_obj:
            error = "Unknown evaluator '{0}'".format(evaluator_name)
            raise ValueError(error)
        else:
            return evaluators.name_to_obj[evaluator_name]()

    def create_population(self, **kwargs):
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
        d = {
            "problem" : self.json(),
            "population" : self.population.json()
        }
        print "Done."

        print "Saving json.",
        with open(fname, "w") as f:
            json.dump(d, f, sort_keys=True, indent=4)
        print "Done."

    def save_image(self):
        fname = self.generation_filename() + ".bmp"
        result = np.zeros_like(self.image)

        print "Creating image {0}:".format(fname)
        print "\tGenerating Image.",
        for individual in self.population.individuals():
            representation = individual.create_mask(self)
            for (y, x) in np.argwhere(representation.any(axis=-1)):
                result[y][x] = representation[y][x]
        print "Done"

        print "\tSaving Image",
        result = Image.fromarray(result, mode="RGB")
        result.save(fname)
        print "Done."

    def generation_filename(self):
        root = os.path.basename(self.image_name)
        padded_generation = str(self.current_generation).zfill(7)

        return "{root}_gen_{padded_generation}".format(
            root=root,
            padded_generation=padded_generation,
        )

    def json(self):
        return {
            "image_name" : self.image_name,
            "image_md5" : self.image_md5,
            "current_generation" : self.current_generation,
            "generations" : self.generations,
            "picture_every" : self.picture_every,
            "evaluator_name" : self.evaluator.__class__.__name__,
            "population_name" : self.population.__class__.__name__,
        }

if __name__ == "__main__":
    problem = Problem(image_name="first.jpg")
    # problem = Problem(snapshot_name="first.jpg")
    problem.run()
    # problem.next_generation()
