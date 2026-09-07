"""Packaged sample models, available in both checkouts and installed wheels."""
from importlib.resources import as_file, files

from .application import load_model


def load_benchy():
    resource = files("treesupport").joinpath("assets", "3DBenchy.stl")
    with as_file(resource) as path:
        return load_model(path)
