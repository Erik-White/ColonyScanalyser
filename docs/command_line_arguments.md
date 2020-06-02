# Command Line Arguments
A list of optional arguments, and their default values, that affect the way ColonyScanalyser is run. Some arguments have a 'shorthand' version that saves having to type the full command. Shorthand arguments are prefixed by a single dash `-`, while normal arguments are prefixed by two dashes `--`

The only required argument is the path to the folder containing the image files you wish to analyse. Optional arguments can be added after the folder path e.g.
```
scanalyser /path/to/images --use_saved false
```
You can add as many optional arguments as you like:
```
scanalyser /path/to/images --use_saved false -mp true -dpi 300 --plate_lattice 2 3
```
### Help
A full list of available arguments, along with their default values
```
-h
--help
```
### Image density
The image density your scanner uses, this can usually be found in your scanner settings. It is important to set this correctly as it enables the program to acurately convert the plate size in millimeters to pixels.

- input: integer
```
-dpi
--dots_per_inch
```
### Image formats
Displays the currently supported image formats.

```
--image_formats
```
### Multiprocessing
This technique utilises all of the available processors that your computer has to analyse images in parallel. Since most computers now have at least 2 or 4 processors, this can greatly reduce the time needed to process a set of images.

This technique is however quite resource intensive for your computer so you may wish to disable it.

- input: boolean
```
-mp
--multiprocessing
```
### Plot images output
The level of detail required when saving plot images after analysis. At the default level (`1`), a few summary plots are saved to give a quick overview of the data. If the output level is increased, individual plots for each plate will be saved.

At the highest level (currently `4`), animations of plate images are created in `gif` format. These may take several minutes to create and the process can be quite resource intensive.

- input: boolean
```
-p
--plots
```
### Plate edge cut
The radius, as a percentage of the plate diameter, to exclude from the edge of the plate image. This ensures that the image is clear of reflections, shadows and writing that are typically present near the edge of the plate image.

- input: integer
```
--plate_edge_cut
```
### Plate labels
A list of labels to identify each plate. The label is used in file names and the plate map.

Plates are ordered from top left, in rows, and labels must be provided in that order.

Labels are separated with spaces. To use a space within a label, wrap that label in quotes

Example: `--plate_labels first second third "label with spaces" fifth sixth`

- input: list
```
--plate_labels
```
### Plate holder shape
The layout of the plates in the image in rows and columns. The default is `3` rows and `2` columns.

A square grid of 9 plates would be entered as `--plate_lattice 3 3`

- input: integer
```
--plate_lattice
```
### Plate size
The diameter of the plates used, in millimeters. It is important to set this correctly otherwise the plates may be located incorrectly in the images.

- input: integer
```
--plate_size
```
### Cached data
The package saves a compressed serialised version of its output, along with the uncompressed CSV data. This allows it to quickly generate the CSV files and plot images again, without the need for reanalysing the original images. This is disabled by default to prevent confusing situation where outdated information is output from new or altered image sets.

- input: boolean
```
--use_cached_data
```
### Information output
The level of information output to the command line. Default level is `1`, increase to see more information. Output can be silenced with `0`

- input: integer
```
-v
--verbose
```
### Package version
Displays the version number of the installed ColonyScanalyser package

```
--version
```

