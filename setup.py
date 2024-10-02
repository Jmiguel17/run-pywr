from setuptools import setup, find_packages

setup(
    name='run-pywr',
    version='0.1',
    description='module for run pywr',
    url='',
    author='Jose M. Gonzalez',
    author_email='jgonzalez@nexsys-analytics.com',
    packages=find_packages(),
    package_data={
        'run_pywr': ['json/*.json'],
    },
    entry_points={
        'console_scripts': ['run_pywr=run_pywr.cli:start_cli'],
    }
)