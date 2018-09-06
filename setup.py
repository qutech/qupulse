from setuptools import setup, find_packages
import sys

if sys.version_info < (3, 3):
    sys.stderr.write('ERROR: You need Python 3.3 or later '
                     'to install the qupulse package.\n')
    exit(1)

if sys.version_info < (3, 5):
    requires_typing = ['typing==3.5.0']
else:
    requires_typing = []

packages = [package for package in find_packages()
            if package.startswith('qupulse')] + ['qctoolkit']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='qupulse',
      version='0.1.1',
      description='A Quantum compUting PULse parametrization and SEquencing framework',
      long_description=long_description,
      long_description_content_type="text/markdown",

      author='Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University',
      package_dir={'qupulse': 'qupulse', 'qctoolkit': 'qctoolkit'},
      packages=packages,
      python_requires='>=3.3',
      tests_require=['pytest'],
      install_requires=['sympy>=1.1.1', 'numpy', 'cached_property', 'gmpy2'] + requires_typing,
      extras_require={
          'testing': ['pytest'],
          'plotting': ['matplotlib'],
          'faster expressions': ['numexpr'],
          'VISA': ['pyvisa'],
          'tabor instruments': ['pytabor>=1.0.1', 'teawg']
      },
      test_suite="tests",
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      url="https://github.com/qutech/qupulse",
)
