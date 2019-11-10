#Command Line Arguments
A list of optional arguments, and their default values, that affect the way ColonyScanalyser is run. Some arguments have a 'shorthand' version that saves having to type the full command. Shorthand arguments are prefixed by a single dash `-`, while normal arguments are prefixed by two dashes `--`

The only required argument is the path to the folder containing the image files you wish to analyse. Optional arguments can be added after the folder path e.g.
```
scanalyser /path/to/images --use_saved false
```
You can add as many optional arguments as you like:
```
scanalyser /path/to/images --use_saved false -mp true -dpi 2540 --plate_lattice 2 3
```
###Help
A full list of available arguments, along with their default values
```
-h
--help
```
###Information output
The level of information output to the command line. Default level is `1`, increase to see more information. Output can be silenced with `0`
```
-v
--verbose
```
###Image density
The image density your scanner uses, this can usually be found in your scanner settings. It is important to set this correctly as it enables the program to acurately convert the plate size in millimeters to pixels.
```
-dpi
--dots_per_inch
```
###Plate size
The diameter of the plates used, in millimeters. It is important to set this correctly otherwise the plates may be located incorrectly in the images.
```
--plate_size
```
###Plate lattice
The layout of the plates in the image in rows and columns. The default is `3` rows and `2` columns.

A square grid of 9 plates would be entered as `--plate_lattice 3 3`
```
--plate_lattice
```
###Plot images output
The level of detail required when saving plot images after analysis. At the default level (`1`), a few summary plots are saved to give a quick overview of the data. If the output level is increased, individual plots for each plate will be saved.

Warning: increasing the number of plots can greatly increase the time taken for the image analysis
```
--save_plots
```
###Cached data
The package saves a compressed version of its output, along with the uncompressed CSV data. This allows it to quickly generate the CSV files and plot images again, without the need for analysing all the original images again. If you would prefer to start the analysis from the beginning, this can be disabled.
```
--use_saved
```
###Multiprocessing
This technique utilises all of the available processors that your computer has to analyse images in parallel. Since most computers now have at least 2 or 4 processors, this can greatly reduce the time needed to process a set of images.

This technique is however quite resource intensive for your computer so you may wish to disable it.
```
-mp
--multiprocessing
```

