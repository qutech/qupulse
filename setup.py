from setuptools import setup, find_packages
import sys
import os
import re


def extract_version(version_file):
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


if sys.version_info < (3, 5):
    sys.stderr.write('ERROR: You need Python 3.5 or later '
                     'to install the qupulse package.\n')
    exit(1)

packages = [package for package in find_packages()
            if package.startswith('qupulse')] + ['qctoolkit']

with open("README.md", "r") as fh:
    long_description = fh.read()

with open(os.path.join('qupulse', '__init__.py'), 'r') as init_file:
    init_file_content = init_file.read()

setup(name='qupulse',
      version=extract_version(init_file_content),
      description='A Quantum compUting PULse parametrization and SEquencing framework',
      long_description=long_description,
      long_description_content_type="text/markdown",

      author='Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University',
      package_dir={'qupulse': 'qupulse', 'qctoolkit': 'qctoolkit'},
      packages=packages,
      python_requires='>=3.5',
      install_requires=['sympy>=1.1.1', 'numpy', 'cached_property'],
      extras_require={
          'plotting': ['matplotlib'],
          'VISA': ['pyvisa'],
          'tabor-instruments': ['pytabor>=1.0.1', 'teawg'],
          'Faster-fractions': ['gmpy2']
      },
      test_suite="tests",
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      url="https://github.com/qutech/qupulse",
)
