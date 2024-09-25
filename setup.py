import os
import re
import sys

from setuptools import find_packages, setup

_deps = [
    "diffusers",
    "transformers",
    "huggingface-hub",
    "requests",
    "torch",
    "jax",
    "natsort",
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}

def read_version():
    with open(os.path.join(os.path.dirname(__file__), 'Version.txt')) as version_file:
        return version_file.read().strip()


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


version_range_max = max(sys.version_info[1], 10) + 1

setup(
    name="auto_diffusers",
    version=read_version(),
    description="Customized diffusers with model search and other functions.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="diffusers model search deep learning diffusion jax pytorch stable diffusion",
    license="BSD 3-Clause License",
    author="suzukimain",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ]
    + [f"Programming Language :: Python :: 3.{i}" for i in range(8, version_range_max)]  
)
