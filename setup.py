from distutils.core import setup

setup(
    name='procrustes',
    version='1.0',
    packages=['procrustes', 'procrustes.hungarian', 'procrustes.procrustes', 'procrustes.procrustes.test'],
    url='',
    test_suite='unittest',
    license='MIT',
    author='Jonathan La, Farnaz Zadeh',
    author_email='lajd@mcmaster.ca',
    description='A package for basic procrustes problems'
)
