# -*- coding: UTF-8 -*-
#! /usr/bin/python

import sys
import os
from setuptools import find_packages
from numpy.distutils.core import setup
from numpy.distutils.core import Extension
import mlhiphy

NAME    = 'mlhiphy'
VERSION = mlhiphy.__version__
AUTHOR  = 'A. Ratnani, K. Harsha, A. Grundner, K. Wang'
EMAIL   = 'ratnaniahmed@gmail.com'
URL     = 'http://www.ahmed.ratnani.org'
DESCR   = 'Machine Learning for Hidden Physics'
KEYWORDS = ['PDE', 'Gaussian Process']
LICENSE = "LICENSE.txt"

setup_args = dict(
    name             = NAME,
    version          = VERSION,
    description      = DESCR,
    long_description = open('README.md').read(),
    author           = AUTHOR,
    author_email     = EMAIL,
    license          = LICENSE,
    keywords         = KEYWORDS,
    url              = URL,
)

# ...
packages = find_packages()
# ...

# ...
install_requires = ['numpy', 'scipy', 'sympy']
dependency_links = []
# ...

# ...
def setup_package():
    setup(packages=packages,
          install_requires=install_requires,
          include_package_data=True,
          zip_safe=True,
          dependency_links=dependency_links,
          **setup_args)
# ....
# ..................................................................................
if __name__ == "__main__":
    setup_package()
