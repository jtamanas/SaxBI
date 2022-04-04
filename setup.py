from __future__ import absolute_import, division, print_function

# Standard imports
import sys
import glob, os
import pdb
from setuptools import setup, find_packages

# Begin setup
setup_keywords = dict()
setup_keywords["name"] = "saxbi"
setup_keywords["description"] = "JAX implementation of Sequential Neural Likelihood Estimation (SNLE) and Sequential Neural Ratio Estimation (SNRE) simulation-based inference algorithms"
setup_keywords["author"] = "John Tamanas"
setup_keywords["author_email"] = "jtamanas@gmail.com"
setup_keywords["license"] = "BSD"
setup_keywords["url"] = "https://github.com/jtamanas/saxbi"
setup_keywords["version"] = "0.0.dev0"
# Use README.rst as long_description.
setup_keywords["long_description"] = ""
if os.path.exists("README.md"):
    with open("README.md") as readme:
        setup_keywords["long_description"] = readme.read()
setup_keywords["provides"] = [setup_keywords["name"]]
setup_keywords["requires"] = ["Python (>3.9.0)"]
setup_keywords["setup_requires"] = [
    "jax",
    "jaxlib",
    "numpy",
]
setup_keywords["install_requires"] = [
    "torch",
    "optax", 
    "flax",
    "numpyro",
    "corner",
    "scikit-learn",
    "tqdm",
]
setup_keywords["zip_safe"] = False
setup_keywords["use_2to3"] = False
setup_keywords["packages"] = find_packages()
setup_keywords["setup_requires"] = ["pytest-runner"]
setup_keywords["tests_require"] = ["pytest"]

setup(**setup_keywords)
