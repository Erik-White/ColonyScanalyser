import io
import re
from pathlib import Path
from setuptools import setup, find_namespace_packages


def read(*names, **kwargs):
    with io.open(
        Path.joinpath(Path(__file__).parent, *names),
        encoding = kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


# Combine the readme and changelog to form the long_description
long_description = "%s\n%s" % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub("", read("README.md")),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("docs/CHANGELOG.md"))
    )


setup(
    python_requires = ">=3.7",
    name = "colonyscanalyser",
    version = "0.6.2",
    description = "An image analysis tool for measuring microorganism colony growth",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Erik-White/ColonyScanalyser/",
    author = "Erik White",
    author_email = "",
    license = "GPL-3.0",
    packages = find_namespace_packages(where = "src", exclude = ["tests", "tests.*"]),
    package_dir = {"": "src"},
    zip_safe = False,
    install_requires = [
        "numpy >= 1.18",
        "matplotlib",
        "scikit-image >= 0.17",
        "imreg_dft",
        "webcolors",
        # Ensure metadata is available on Python <3.8
        "importlib-metadata ~= 1.0 ; python_version < '3.8'"
    ],
    extras_require = {
        "dev": [
            "flake8",
            "check-manifest",
            "pytest",
            "pytest-cov"
        ],
        "test": [
            "flake8",
            "pytest",
            "pytest-cov"
        ],
    },
    entry_points={
        'console_scripts': [
            'colonyscanalyser = colonyscanalyser.main:main',
            'scanalyser = colonyscanalyser.main:main',
        ],
    },
)