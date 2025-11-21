import os
import re
import sys

from setuptools import find_packages, setup

_deps = [
    "diffusers>=0.32.0",
    "transformers>=4.41.2",
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


version_range_max = max(sys.version_info[1], 10) + 1

setup(
    name="auto_diffusers",
    version="2.0.29",
    description="diffusers with search engine",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="huggingface civitai diffusers model search deep learning diffusion pytorch stable diffusion",
    license="Apache 2.0 License",
    author="suzukimain",
    author_email="subarucosmosmain@gmail.com",
    url="https://github.com/suzukimain/auto_diffusers",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["py.typed"]},
    python_requires=">=3.8.0",
    include_package_data=True,
    install_requires=list(_deps),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ]
    + [f"Programming Language :: Python :: 3.{i}" for i in range(8, version_range_max)]  
)
