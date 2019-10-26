from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    python_requires = ">3.7",
    name = "colonyscanalyser",
    version = "0.2.0",
    description = "An image analysis tool for measuring microorganism colony growth",
    long_description = long_description,
    url = "https://github.com/Erik-White/ColonyScanalyser/",
    author = "Erik White",
    author_email = "",
    license = "GPL-3.0",
    packages =  find_packages(),
    zip_safe = False,
    install_requires = [
        "numpy",
        "matplotlib",
        "scikit-image >= 0.15"
    ],
    extras_require = {
        "dev": [
            "check-manifest"
            "pytest",
            "pytest-cov"
            ],
        "test": [
            "pytest",
            "pytest-cov"
        ],
    },
    entry_points={
        'console_scripts': [
            'colonyscanalyser = colonyscanalyser.colonyscanalyser:main',
            'scanalyser = colonyscanalyser.colonyscanalyser:main',
        ],
    },
)