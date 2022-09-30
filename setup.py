from setuptools import (  # type: ignore
    setup,  # type: ignore
    find_packages,  # type: ignore
)

version = "0.0.1"

scripts = ["gps/cli/gps"]

install_requires = [
    "sklearn",
    "numpy",
    "numba",
    "torch",
    "pytorch_lightning",
    "torchvision",
    "imblearn",
    "xgboost==1.5.0"
]

setup(
    name="gps",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    version=version,
    scripts=scripts,
    license="",
    author="Aaron Scott",
    author_email="aaron.scott@med.lu.se",
)
