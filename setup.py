"""Copyright (c) Dreamfold."""
#!/usr/bin/env python

import os

from setuptools import find_packages, setup

version_py = os.path.join(os.path.dirname(__file__), "FoldFlow", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()
setup(
    name="FoldFlow",
    version=version,
    description="Fold Flow on SO3",
    author="Dreamfold",
    install_requires=["torch", "pot", "numpy", "torchdyn"],
    packages=find_packages(),
)