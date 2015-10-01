from setuptools import setup
import sys

if sys.version_info < (3, 5):
    requires_typing = ['typing==3.5.0b1']
else:
    requires_typing = []

subpackages = ['pulses','utils']
packages = ['qctoolkit'] + ['qctoolkit.' + subpackage for subpackage in subpackages]

setup(name='qctoolkit',
    version='0.1',
    description='Quantum Computing Toolkit',
    author='qutech',
    package_dir = {'qctoolkit': 'src'},
    packages=packages,
    tests_require=['pytest'],
    install_requires= ['py_expression_eval'] + requires_typing,
    extras_require={
        'testing' : ['pytest'],
        'plotting' : ['matplotlib'],
        'faster expressions' : ['numexpr']
    },
    test_suite="tests",
)