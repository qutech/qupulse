from setuptools import setup
import sys

if sys.version_info < (3, 5):
    requires_typing = ['typing==3.5.0']
else:
    requires_typing = []

setup(name='qctoolkit',
    version='0.1',
    description='Quantum Computing Toolkit',
    author='qutech',
    package_dir = {'': 'src'},
    packages=['pulses', 'utils'],
    tests_require=['pytest'],
    install_requires= ['py_expression_eval'] + requires_typing,
    extras_require={
        'testing' : ['pytest'],
        'plotting' : ['matplotlib'],
        'faster expressions' : ['numexpr']
    },
    test_suite="tests",
)
