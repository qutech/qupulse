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

setup(name='qupulse',
      version='0.1',
      description='A Quantum compUting PULse parametrization and SEquencing framework',
      author='Quantum Technology Group and Chair of Software Engineering, RWTH Aachen University',
      package_dir={'qupulse': 'qupulse', 'qctoolkit': 'qctoolkit'},
      packages=packages,
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
)
