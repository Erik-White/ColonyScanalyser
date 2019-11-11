# Example output
A few simple plots are saved to allow quick comparison of the different plates. Increasing the `save_plots` argument will increase the number and detail of plots that are output, see the [command line arguments](command_line_arguments.md) for more detail.

Data such as colony area, growth rate, time of appearance etc is output to a set of CSV files.
## Plate identification
Plates are numbered sequentially, starting from the top left and proceeding in rows. On a 3 x 2 lattice the plate numbers will be:
```
1   2

3   4

5   6
```

<img align="right" src="/images/time_of_appearance_small.png">

## Plots
At the default level, a few comparison plots are output to the `plots` folder in your image folder. Theses allow quick verification and comparison of the data.

Increasing the level of `save_plots` will output plots for each individual plate.

At the highest level of `save_plots`, an image plot showing colony segmentation at each timepoint, for each plate will be output. This may be useful for viewing how the image segmentation criteria have identified objects as colonies.

Note: extra filtering is applied after image segmentation so not all objects that appear in the segmented image plots will be included in the aggregated data. See [colony filtering criteria](colony_filtering.md) for more detail.
## Data
All the data gathered during analysis is output to the `data` folder in your images folder. Data is output as CSV files which is compatible with almost all data packages.

The data is collated in two files for each plate, one with aggregate data for each colony and another with colony data at every image time point:
```
plate1_colonies.csv
plate1_colony_timepoints.csv
plate2_colonies.csv
plate2_colony_timepoints.csv
plate3_colonies.csv
plate3_colony_timepoints.csv
...
```

A single compressed data file, `processed_data.xz`, is also saved. This contains all the data objects from analysis and can be used by the package to quickly recreate the plots and data files.