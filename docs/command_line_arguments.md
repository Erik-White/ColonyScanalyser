# Command Line Arguments
A list of optional arguments, and their default values, that affect the way ColonyScanalyser is run. Some arguments have a 'shorthand' version that saves having to type the full command. Shorthand arguments are prefixed by a single dash `-`, while normal arguments are prefixed by two dashes `--`

The only required argument is the path to the folder containing the image files you wish to analyse. Optional arguments can be added after the folder path e.g.
```
scanalyser /path/to/images --use-cached-data
```
You can add as many optional arguments as you like:
```
scanalyser /path/to/images --use-cached-data -s -d 300 --plate-lattice 2 3
```
### Help
A full list of available arguments, along with their default values
```
-h
--help
```
### Animation
Generate animation and video from plots or plate images. These may take several minutes to create and the process can be quite resource intensive.

```
-a
--animation
```
### Image density
The image density your scanner uses, this can usually be found in your scanner settings. It is important to set this correctly as it enables the program to acurately convert the plate size in millimeters to pixels.

- input: integer
```
-d
--dots-per-inch
```
### Image formats
Displays the currently supported image formats.

```
--image-formats
```
### Plot images output
Prevent plot images being output to disk. Summary plots to give a quick overview of the data and individual plots for each plate are output unless this argument is passed.

```
--no-plots
```
### Plate edge cut
The radius, as a percentage of the plate diameter, to exclude from the edge of the plate image. This ensures that the image is clear of reflections, shadows and writing that are typically present near the edge of the plate image.

- input: integer
```
--plate-edge-cut
```
### Plate labels
A list of labels to identify each plate. The label is used in file names and the plate map.

Plates are ordered from top left, in rows, and labels must be provided in that order.

Labels are separated with spaces. To use a space within a label, wrap that label in quotes

Example: `--plate-labels first second third "label with spaces" fifth sixth`

- input: list
```
--plate-labels
```
### Plate holder shape
The layout of the plates in the image in rows and columns. The default is `3` rows and `2` columns.

A square grid of 9 plates would be entered as `--plate-lattice 3 3`

- input: integer
```
--plate-lattice
```
### Plate size
The diameter of the plates used, in millimeters. It is important to set this correctly otherwise the plates may be located incorrectly in the images.

- input: integer
```
--plate-size
```
### Silence output
Silence all output to the console. Note that this only silences the informative output from the package, warnings or errors may still appear.

```
-s
--silent
```
### Single processing
Disable multiprocessing. By default the program utilises nearly all of the available processors in your system, leaving at least one free for other tasks. This allows image processing in parallel and will greatly reduce the time needed to process a set of images, but is very resource intensive.

This technique is however quite resource intensive for your computer so you may wish to disable it.

```
--single-process
```
### Cached data
The package saves a compressed serialised version of its output, along with the uncompressed CSV data. This allows it to quickly generate the CSV files and plot images again, without the need for reanalysing the original images. This is disabled by default to prevent situations where outdated information is output from new or altered image sets.

```
-u
--use-cached-data
```
### Verbose output
Increases the information output to the command line.

```
-v
--verbose
```
### Package version
Displays the version number of the installed ColonyScanalyser package

```
--version
```

