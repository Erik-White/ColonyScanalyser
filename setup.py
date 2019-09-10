from setuptools import setup

setup(
    python_requires = ">3.7",
    name = "scanlag",
    version = "0.1",
    description = "Runs ScanLag analysis on image files",
    url = "https://github.com/Erik-White/ScanLag/",
    author = "",
    author_email = "",
    license = "GPL-3.0",
    packages =  find_packages(),
    zip_safe = False
    install_requires = [
        "numpy",
        "matplotlib",
        "scikit-image >= 0.16"
    ],
    extras_require = {
        #"dev": ["check-manifest"],
        "test": [
            "mock",
            "pytest",
            "pytest-cov"
        ],
    }
)