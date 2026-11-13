"""Package discovery for setuptools; all metadata lives in pyproject.toml.

Every package under pixelcanvas/ is found automatically (pixelcanvas.clay, pixelcanvas.pictograms and any kit that
grows into a folder later). The runtime folders tools/, scenes/ and skills/ ship inside the wheel as
pixelcanvas._runtime.{tools,scenes,skills} (mapped in [tool.setuptools.package-dir] of pyproject.toml).
"""
from setuptools import find_packages, setup

RUNTIME = ["pixelcanvas._runtime.tools", "pixelcanvas._runtime.scenes", "pixelcanvas._runtime.skills"]

setup(packages=find_packages(include=["pixelcanvas", "pixelcanvas.*"]) + RUNTIME)
