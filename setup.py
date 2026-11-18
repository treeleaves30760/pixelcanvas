"""Package discovery for setuptools; all metadata lives in pyproject.toml.

Finds pixelcanvas and adds pixelcanvas.skill: the skill folder skills/pixelcanvas, mapped
into the package in [tool.setuptools.package-dir] of pyproject.toml so the wheel carries SKILL.md for
`pixelcanvas install-skill`.
"""
from setuptools import find_packages, setup

SKILL = ["pixelcanvas.skill"]

setup(packages=find_packages(include=["pixelcanvas", "pixelcanvas.*"]) + SKILL)
