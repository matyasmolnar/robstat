from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'ROBust STATistical package for data analysis'
LONG_DESCRIPTION = 'ROBust STATistical package for data analysis'

# Setting up
setup(
        name='robstat',
        version=VERSION,
        author='Matyas Molnar',
        author_email='mdm49@cam.ac.uk',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[],
        keywords=['python', 'robust', 'statistics']
)
