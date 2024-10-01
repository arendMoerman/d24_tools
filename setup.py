import os
import pathlib

from setuptools import setup

setup(
    name='d24_tools',
    version='1.0.0',
    author="Arend Moerman",
    install_requires = ["numpy", "matplotlib", "scipy", "decode", "demerge", "dems", "astropy", "xarray"],
    package_dir = {'': 'src'},
    packages=['d24_tools'],
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.8',
)
