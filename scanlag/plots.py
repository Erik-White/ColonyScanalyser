import matplotlib
matplotlib.use("TkAgg") # Required for OSX
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utilities import average_dicts_values_by_key
from plotting import axis_minutes_to_hours


def plot_growth_curve(plate_item, time_points_elapsed, save_path):
    """
    Growth curve, with mean, for a single plate
    """
    fig, ax = plt.subplots()

    # Plot areas for each colony
    colony_growth_curve(ax, plate_item, time_points_elapsed, "Mediumpurple", "Purple")

    lgd = ax.legend(loc = 'center right', fontsize = 8, bbox_to_anchor = (1.25, 0.5))
    save_params = {"format": "png",
          "bbox_extra_artists": (lgd,),
          "bbox_inches": "tight"
          }
    
    plt.ylim(ymin = 0)
    plt.title("Colony growth")
    fig.savefig(str(save_path.joinpath("growth_curve.png")), **save_params)

    # Plot with logarithmic areas
    plt.ylim(ymin = 1)
    ax.set_yscale("log")
    ax.set_ylabel("Colony area [px^2]")
    fig.savefig(str(save_path.joinpath("growth_curve_log.png")), **save_params)

    plt.close()


def plot_growth_curve_all(plate_list, plate_lattice, time_points_elapsed, save_path):
    """
    Compare growth curves on all plates of the plate lattice
    """
    fig, ax = plt.subplots()
    colormap = cm.get_cmap("plasma")

    # Plot areas for each colony
    for plate_item in plate_list.items():
        # Get a color from the colourmap
        cm_plate = colormap(0.2 + (0.65 - 0.2) * (plate_item[0] / len(plate_list.keys())))

        # Add the growth curve plot for this plate
        colony_growth_curve(ax, plate_item, time_points_elapsed, cm_plate)

    lgd = ax.legend(loc = 'center right', fontsize = 8, bbox_to_anchor = (1.25, 0.5))
    save_params = {"format": "png",
          "bbox_extra_artists": (lgd,),
          "bbox_inches": "tight"
          }
    
    plt.ylim(ymin = 0)
    plt.title("Colony growth")
    fig.savefig(str(save_path.joinpath("growth_curve_comparison.png")), **save_params)

    plt.close()


def colony_growth_curve(ax, plate_item, time_points_dict, scatter_color, line_color = None):
    """
    Add a growth curve scatter plot, with mean, to an axis
    """
    plate_id, plate = plate_item
    areas_average = list()

    if line_color is None:
        line_color = scatter_color

    for colony in plate.values():
        # Map areas to a dictionary of all timepoints
        time_points_dict = dict.fromkeys(time_points_dict)
        for timepoint in colony.timepoints.values():
            time_points_dict[timepoint.elapsed_minutes] = timepoint.area
        
        # Store dictionary for averaging, keeping only elements with a value
        areas_average.append(dict(filter(lambda elem: elem[1] is not None, time_points_dict.items())))
            
        # Use zip to return a sorted list of tuples (key, value) from the dictionary
        ax.scatter(*zip(*sorted(time_points_dict.items())),
            color = scatter_color,
            marker = "o",
            s = 1,
            alpha = 0.25,
            )

    # Plot the mean
    areas_averages = average_dicts_values_by_key(areas_average)
    ax.plot(*zip(*sorted(areas_averages.items())),
        color = line_color,
        label = f"Plate {plate_id}",
        linewidth = 2
        )

    # Format x-axis labels as integer hours
    ax.set_xticklabels(axis_minutes_to_hours(ax.get_xticks()))
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("Colony area (pixels)")


def plot_appearance_frequency(plates_dict, time_points_elapsed, save_path, bar = False):
    """
    Compare time of appearance frequency on all plates of the plate lattice
    """
    fig, ax = plt.subplots()
    colormap = cm.get_cmap("plasma")
    
    for plate_item in plates_dict.items():
        if len(plates_dict.keys()) > 1:
            # Get a color from the colourmap
            cm_plate = colormap(0.2 + (0.65 - 0.2) * (plate_item[0] / len(plates_dict.keys())))
        else:
            cm_plate = "Purple"

        # Plot frequency for each time point
        time_of_appearance_frequency(ax, plate_item, time_points_elapsed, cm_plate, bar = bar)

    lgd = ax.legend(loc = 'center right', fontsize = 8, bbox_to_anchor = (1.25, 0.5))
    save_params = {"format": "png",
          "bbox_extra_artists": (lgd,),
          "bbox_inches": "tight"
          }
    
    plt.ylim(ymin = 0)
    plt.title("Time of appearance")
    if bar:
        save_name = "time_of_appearance_bar.png"
    else:
        save_name = "time_of_appearance.png"
    fig.savefig(str(save_path.joinpath(save_name)), **save_params)

    plt.close()


def time_of_appearance_frequency(ax, plate_item, time_points_dict, line_color, bar = False):
    """
    Add a time of appearance frequency bar or line plot to an axis
    """
    plate_id, plate = plate_item

    # Map counts to a dictionary of all timepoints
    time_points_dict = dict()
    for colony in plate.values():
        key = colony.timepoint_first.elapsed_minutes
        if key not in time_points_dict:
            time_points_dict[key] = 0
        time_points_dict[key] += 1

    # Normalise counts to frequency
    time_points_dict = {key: value / len(time_points_dict.keys()) for key, value in time_points_dict.items()}

    # Use zip to return a sorted list of tuples (key, value) from the dictionary
    if not bar:
        ax.plot(*zip(*sorted(time_points_dict.items())),
            color = line_color,
            label = f"Plate {plate_id}",
            alpha = 0.9
            )
    else:
        ax.bar(*zip(*sorted(time_points_dict.items())),
            width = 10,
            color = line_color,
            label = f"Plate {plate_id}",
            alpha = 0.6
            )

    # Format x-axis labels as integer hours
    ax.set_xticklabels(axis_minutes_to_hours(ax.get_xticks()))
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("Frequency")