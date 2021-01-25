from setuptools import (
    setup,
    find_packages
)

version = '0.0.01'

scripts = [
    'gscore/cli/gscore'
]

install_requires = [
    'pandas',
    'sklearn',
    'numpy',
    'tensorflow',
    'matplotlib'
]

setup(
    name='gscore',
    packages=find_packages(),
    install_requires=install_requires,
    version=version,
    scripts=scripts,
    license='',
    author='Aaron Scott',
    author_email='aaron.scott@med.lu.se'
)

