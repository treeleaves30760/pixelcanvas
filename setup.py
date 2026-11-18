"""Package discovery for setuptools; all metadata lives in pyproject.toml.

Finds pixelcanvas and pixelcanvas.easel, and adds pixelcanvas.easel.skill: the skill folder skills/easel, mapped
into the package in [tool.setuptools.package-dir] of pyproject.toml so the wheel carries SKILL.md for
`easel-install-skill`.
"""
from setuptools import find_packages, setup

SKILL = ["pixelcanvas.easel.skill"]

setup(packages=find_packages(include=["pixelcanvas", "pixelcanvas.*"]) + SKILL)
