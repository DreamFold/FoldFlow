"""Copyright (c) Dreamfold."""
#!/usr/bin/env python

import os
from setuptools import setup

version_py = os.path.join(os.path.dirname(__file__), "foldflow", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()

setup(
    name="foldflow",
    packages=["foldflow", "openfold", "ProteinMPNN"],
    package_dir={
        "foldflow": "./foldflow",
        "openfold": "./openfold",
        "ProteinMPNN": "./ProteinMPNN",
        "runner": "./runner",
    },
    install_requires=["torch", "pot", "numpy"],
    version=version,
)
