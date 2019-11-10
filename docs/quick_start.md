# Quick start guide

## Install Python
ColonyScanalyser requires Python version 3.7, and the Pip package manager. Pip is included with Python version 3.4 and up.

There are many [detailed guides on the web](https://docs.python-guide.org/starting/installation/) so I will only cover the most basic cases here.
#### On Mac
I recommend using the [Homebrew package manager](https://brew.sh/) to manage your Python installation.

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
Go to the [Python downloads site](https://www.python.org/downloads/windows/) and grab the latest Python 3.7 installer (currently 3.7.5). Run the installer once it has downloaded and follow the setup instructions.

## Install ColonyScanalyser
Once Python 3.7 is installed we are ready to use the Python package manager, Pip, to install ColonyScanalyser. Open a command line window and install the package:
```
pip3 install colonyscanalyser
```
You should hopefully see a message to say that the ColonyScanalyser package was downloaded and installed successfully.

Note: We use the command `pip3` in case both Python 2 & 3 are installed on the system. In which case `pip` will usually default to installing packages for Python 2

## Analyse your images
Now that we have installed the ColonyScanalyser package we can run it from the command line. The package is run with the command `scanalyser`, followed by any arguments

Try running the command with the `--help` or `-h` argument to see a list of available arguments and their default values:
```
scanalyser --help
```
The only argument you normally need is the folder path to your images:
```
scanalyser images_folder/path
```
If this path contains spaces you must enclose it in quotes:
```
scanalyser "/images folder/path"
```
If you have entered a correct folder path you will see some messages from ColonyScanalyser as it begins to process and analyse your data.

