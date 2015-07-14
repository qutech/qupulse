from setuptools import setup

setup(name='qctoolkit',
    version='0.1',
    description='Quantum Computing Toolkit',
    author='qutech',
    package_dir = {'': 'src'},
    packages=['pulses', 'utils'],
    tests_require=['pytest'],
    extras_require={
        'testing' : ['pytest'],
    },
    test_suite="tests",
)