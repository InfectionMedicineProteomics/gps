from setuptools import (  # type: ignore
    setup,  # type: ignore
    find_packages  # type: ignore
)

version = '0.0.01'

scripts = [
    'gscore/cli/gscore'
]

install_requires = [
    'sklearn',
    'numpy',
    #'tensorflow',
    'matplotlib',
    'seaborn',
    'pomegranate==0.14.0'
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

