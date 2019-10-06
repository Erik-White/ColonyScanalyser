import matplotlib
matplotlib.use("TkAgg") # Required for OSX
import matplotlib.pyplot as plt


def plot_colony_growth_curve(plate_colonies, time_points_elapsed, save_path):
    """
    Growth curve, with mean, for a single plate
    """
    import utilities

    fig, ax = plt.subplots()

    # Plot areas for each colony
    areas_average = list()
    for colony in plate_colonies.values():
        # Map areas to a dictionary of all timepoints
        time_points_dict = dict.fromkeys(time_points_elapsed)
        for timepoint in colony.timepoints.values():
            time_points_dict[timepoint.elapsed_minutes] = timepoint.area
        
        # Store dictionary for averaging, keeping only elements with a value
        areas_average.append(dict(filter(lambda elem: elem[1] is not None, time_points_dict.items())))
            
        # Use zip to return a sorted list of tuples (key, value) from the dictionary
        ax.scatter(*zip(*sorted(time_points_dict.items())),
            color = "Purple",
            marker = "o",
            s = 4,
            #label = str(colony_id),
            alpha = 0.1
            )

    # Plot the mean
    areas_averages = utilities.average_dicts_values_by_key(areas_average)
    ax.plot(*zip(*sorted(areas_averages.items())),
        color = "Aqua",
        label = "Mean area",
        linewidth = 1.5
        )

    # Format x-axis labels as integer hours
    ax.set_xticklabels(["{:.0f}".format(x // 60) for x in ax.get_xticks()])

    lgd = ax.legend(loc = 'center right', fontsize = 8, bbox_to_anchor = (1.25, 0.5))
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("Colony area (pixels)")
    plt.title("Colony growth")

    save_params = {"format": "png",
          "bbox_extra_artists": (lgd,),
          "bbox_inches": "tight"
          }
    
    plt.ylim(ymin = 0)
    fig.savefig(str(save_path.joinpath("colony_growth_curve.png")), **save_params)

    # Plot with logarithmic areas
    plt.ylim(ymin = 1)
    ax.set_yscale("log")
    ax.set_ylabel("Colony area [px^2]")
    fig.savefig(str(save_path.joinpath("colony_growth_curve_log.png")), **save_params)

    plt.close()