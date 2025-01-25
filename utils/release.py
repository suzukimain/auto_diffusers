# Based on diffusers/utils/release.py

import argparse
import os
import re

import packaging.version


REPLACE_PATTERNS = {
    "init": (re.compile(r'^__version__\s+=\s+"([^"]+)"\s*$', re.MULTILINE), '__version__ = "VERSION"\n'),
    "setup": (re.compile(r'^(\s*)version\s*=\s*"[^"]+",', re.MULTILINE), r'\1version="VERSION",'),
}
REPLACE_FILES = {
    "init": "src/diffusers/__init__.py",
    "setup": "setup.py",
}
README_FILE = "README.md"


def update_version_in_file(fname, version, pattern):
    """Update the version in one file using a specific pattern."""
    with open(fname, "r", encoding="utf-8", newline="\n") as f:
        code = f.read()
    re_pattern, replace = REPLACE_PATTERNS[pattern]
    replace = replace.replace("VERSION", version)
    code = re_pattern.sub(replace, code)
    with open(fname, "w", encoding="utf-8", newline="\n") as f:
        f.write(code)


def global_version_update(version):
    """Update the version in all needed files."""
    for pattern, fname in REPLACE_FILES.items():
        update_version_in_file(fname, version, pattern)


def get_version():
    """Reads the current version in the __init__."""
    with open(REPLACE_FILES["init"], "r") as f:
        code = f.read()
    default_version = REPLACE_PATTERNS["init"][0].search(code).groups()[0]
    return packaging.version.parse(default_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update_version", action="store_true", help="Update version number.")
    args = parser.parse_args()
    global_version_update(version=args.update_version)