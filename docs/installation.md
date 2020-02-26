# Installation
## Python
ColonyScanalyser requires Python version 3.7 or greater, and the Pip package manager. Pip is included with Python version 3.4 and up.

There are many [detailed guides on the web](https://docs.python-guide.org/starting/installation/) so I will only cover the most basic cases here.
#### On Mac
The [Homebrew package manager](https://brew.sh/) is recommended for managing your Python installation.

Once Homebrew is installed, simply open terminal and run the command
```
brew install python
```
#### On Linux
Python may come bundled with your distribution, otherwise it will be available through your package manager e.g.
```
sudo apt-get install python3.7
```
#### On Windows
Go to the [Python downloads site](https://www.python.org/downloads/windows/) and grab the latest Python installer. Run the installer once it has downloaded and follow the setup instructions.

## ColonyScanalyser
Use the Python package manager, Pip, to install the [ColonyScanalyser package](https://pypi.org/project/colonyscanalyser/):
```
pip3 install colonyscanalyser
```
A message should hopefully indicate that the ColonyScanalyser package was downloaded and installed successfully.

Note: We use the command `pip3` in case both Python 2 & 3 are installed on the system. In which case `pip` will usually default to installing packages for Python 2

## Virtual environments
There are a number of benefits to setting up self-contained virtual environments instead of installing system-wide packages:

- Creates a separate and controlled work environment
- Easily manages complex interdependencies

We use Pipenv which is a combined environment and package manager. Install it with Homebrew:
```
brew install pipenv
```
Then create a new Python 3 environment in a workspace folder and install the ColonyScanalyser package:
```
cd /my/example/workspace
pipenv --three
pipenv install colonyscanalyser
```
Then run the ColonyScanalyser package:
```
pipenv run scanalyser /path/to/images
```

[More detailed instructions](https://packaging.python.org/tutorials/managing-dependencies/) can be found on the Python packaging site.