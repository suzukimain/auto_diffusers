import os
import re
import sys
from distutils.core import Command

from setuptools import find_packages, setup


with open("./requirements.txt", "r", encoding="utf-8") as _file:
    _deps = _file.read().splitlines()

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


version_range_max = max(sys.version_info[1], 10) + 1

setup(
    name="auto_diffusers",
    version="0.2.0",
    description="Customized diffusers with model search and other functions.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="Added the ability to search models and more to diffusers",
    license="BSD 3-Clause License",
    author="suzukimain(https://github.com/suzukimain)",
    author_email="subarucosmosmain@gmail.com",
    url="https://github.com/suzukimain/auto_diffusers",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["py.typed"]},
    include_package_data=True,
    install_requires=list(_deps),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ]
    + [f"Programming Language :: Python :: 3.{i}" for i in range(8, version_range_max)]  
)
