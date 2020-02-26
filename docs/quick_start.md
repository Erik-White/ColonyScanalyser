# Quick start guide

## Install Python
The most basic requirements for the package are simply Python with its package manager

- Python >= 3.7
- Pip

The [installation guide](installation.md) has more information on getting Python.

## Install ColonyScanalyser
Once Python is installed the Python package manager, Pip, can be used to install ColonyScanalyser. Open a command line window and install the package:
```
pip3 install colonyscanalyser
```

## Analyse your images
Now that the ColonyScanalyser package is installed, we can run it from the command line. The package is run with the command `scanalyser`, followed by any arguments

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
See the [command line arguments](command_line_arguments.md) page for a full list and further details.

If you have entered a correct folder path you will see some messages from ColonyScanalyser as it begins to process and analyse your data.

