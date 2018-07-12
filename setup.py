from setuptools import setup
import sys

if sys.version_info < (3, 3):
    sys.stderr.write('ERROR: You need Python 3.3 or later '
                     'to install the qctoolkit package.\n')
    exit(1)

if sys.version_info < (3, 5):
    requires_typing = ['typing==3.5.0']
else:
    requires_typing = []

subpackages = ['pulses', 'utils', 'hardware']
packages = ['qctoolkit'] + ['qctoolkit.' + subpackage for subpackage in subpackages]

setup(name='qctoolkit',
      version='0.1',
      description='Quantum Computing Toolkit',
      author='qutech',
      package_dir={'qctoolkit': 'qctoolkit'},
      packages=packages,
      tests_require=['pytest'],
      install_requires=['sympy==1.1.1', 'numpy'] + requires_typing,
      extras_require={
          'testing': ['pytest'],
          'plotting': ['matplotlib'],
          'faster expressions': ['numexpr'],
          'VISA': ['pyvisa'],
          'tabor instruments': ['pytabor>=1.0.1', 'teawg']
      },
      test_suite="tests",
)
