import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .utilities import average_dicts_values_by_key
from .plotting import rc_to_xy, axis_minutes_to_hours


def plot_colony_map(plate_image, plate_coordinates, plate_colonies, save_path, edge_cut = 0):
    """
    Saves original plate image with overlaid plate and colony IDs

    :param plate_image: the final timepoint image of all plates
    :param plate_coordinates: a dictionary of centre and radii tuples
    :param plate_colonies: a dictionary of Colony objects
    :param save_path: a path object
    :returns: a file path object if the plot was saved sucessfully
    """
    from matplotlib import rcParams

    # Calculate the image size in inches
    dpi = rcParams['figure.dpi']
    height, width, depth = plate_image.shape
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure that takes up the full size of the image
    fig = plt.figure(figsize = figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(plate_image)

    for plate_id, plate in plate_colonies.items():
        (center_y, center_x), plate_radius = plate_coordinates[plate_id]

        # Colony coordinates are relative to individual plate images
        # Calculate a correction factor to allow plotting on the original image
        offset_y = center_y - plate_radius + edge_cut
        offset_x = center_x - plate_radius + edge_cut

        # Label plates
        ax.annotate(
            f"Plate #{plate_id}".upper(),
            (center_x, center_y - plate_radius - (edge_cut * 1.4)),
            xycoords = "data",
            horizontalalignment = "center",
            verticalalignment = "center",
            fontsize = "40",
            backgroundcolor = "black",
            color = "white"
        )

        # Mark the detected boundary of the plate
        plate_circle = plt.Circle(
            (center_x, center_y),
            radius = plate_radius,
            facecolor = "none",
            edgecolor = "purple",
            linewidth = "2.5",
            linestyle = "-",
            label = "Detected plate boundary"
        )
        ax.add_artist(plate_circle)

        # Mark the measured area of the plate
        plate_circle_measured = plt.Circle(
            (center_x, center_y),
            radius = plate_radius - edge_cut,
            facecolor = "none",
            edgecolor = "white",
            linewidth = "1.5",
            linestyle = "--",
            label = "Colony detection area"
        )
        ax.add_artist(plate_circle_measured)

        # Mark colony centres and ID numbers
        for colony in plate.values():
            x, y = rc_to_xy(colony.center)
            x = offset_x + x
            y = offset_y + y
            radius = colony.timepoint_last.diameter / 2

            ax.annotate(
                "+",
                (x, y),
                xycoords = "data",
                color = "red",
                horizontalalignment = "center",
                verticalalignment = "center",
                fontsize = "x-small"
            )
            ax.annotate(
                colony.id,
                (x, y),
                xycoords = "data",
                xytext = (radius * 0.7, -radius * 0.9),
                textcoords = "offset pixels",
                horizontalalignment = "left",
                verticalalignment = "top",
                alpha = 0.85,
                backgroundcolor = "black",
                color = "white",
                fontsize = "small"
            )
            # Mark the calculated radius of the colony
            colony_circle = plt.Circle(
                (x, y),
                radius = radius,
                facecolor = "none",
                edgecolor = "red",
                alpha = 0.75,
                linewidth = "0.65",
                linestyle = "-",
                label = "Colony area at final measurement"
            )
            ax.add_artist(colony_circle)

    plt.legend(
        handles = [plate_circle, plate_circle_measured, colony_circle],
        loc = "lower center",
        facecolor = "lightgray",
        shadow = "true",
        fontsize = "18"
    )

    image_path = "plate_map.png"
    save_path = save_path.joinpath(image_path)
    try:
        plt.savefig(str(save_path), format = "png")
    except Exception:
        save_path = None
    finally:
        plt.close()
        return save_path


def plot_plate_segmented(plate_image, segmented_image, date_time, save_path):
    """
    Saves processed plate images and corresponding segmented data plots

    :param plate_image: a black and white image as a numpy array
    :param segmented_image: a segmented and labelled image as a numpy array
    :param date_time: a datetime object
    :param save_path: a path object
    :returns: a file path object if the plot was saved sucessfully
    """
    from skimage.measure import regionprops

    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(plate_image)
    # Set colour range so all colonies are clearly visible and the same colour
    ax[1].imshow(segmented_image, vmax = 1)

    # Place maker labels on colonies
    for rp in regionprops(segmented_image):
        ax[1].annotate(
            "+",
            rc_to_xy(rp.centroid),
            xycoords = "data",
            color = "red",
            horizontalalignment = "center",
            verticalalignment = "center"
            )

    plt.suptitle(f"Plate time point {date_time.strftime('%Y/%m/%d %H:%M')}")
    image_path = f"time_point_{date_time.strftime('%Y%m%d')}_{date_time.strftime('%H%M')}.png"
    save_path = save_path.joinpath(image_path)
    try:
        plt.savefig(str(save_path), format = "png")
    except Exception:
        save_path = None
    finally:
        plt.close()
        return save_path


def plot_growth_curve(plates_dict, time_points_elapsed, save_path):
    """
    Growth curves for either a single plate, or all plates on the lattice
    """
    _, ax = plt.subplots()
    colormap = cm.get_cmap("plasma")

    for plate_item in plates_dict.items():
        if len(plates_dict) > 1:
            # Get a color from the colourmap
            cm_scatter = colormap(0.2 + (0.65 - 0.2) * (plate_item[0] / len(plates_dict)))
            cm_line = None
        else:
            cm_scatter = "Mediumpurple"
            cm_line = "Purple"

        # Add the growth curve plot for this plate
        growth_curve(ax, plate_item, time_points_elapsed, cm_scatter, cm_line)

    lgd = ax.legend(loc = 'center right', fontsize = 8, bbox_to_anchor = (1.25, 0.5))
    save_params = {
        "format": "png",
        "bbox_extra_artists": (lgd,),
        "bbox_inches": "tight"
        }

    plt.ylim(ymin = 0)
    plt.title("Colony growth")
    plt.savefig(str(save_path.joinpath("growth_curve.png")), **save_params)

    plt.close()


def growth_curve(ax, plate_item, time_points_elapsed, scatter_color, line_color = None):
    """
    Add a growth curve scatter plot, with mean, to an axis
    """
    plate_id, plate = plate_item
    areas_average = list()

    if line_color is None:
        line_color = scatter_color

    for colony in plate.values():
        # Map areas to a dictionary of all timepoints
        time_points_dict = dict.fromkeys(time_points_elapsed)
        for timepoint in colony.timepoints.values():
            time_points_dict[timepoint.elapsed_minutes] = timepoint.area

        # Store dictionary for averaging, keeping only elements with a value
        areas_average.append(dict(filter(lambda elem: elem[1] is not None, time_points_dict.items())))

        # Use zip to return a sorted list of tuples (key, value) from the dictionary
        ax.scatter(
            *zip(*sorted(time_points_dict.items())),
            color = scatter_color,
            marker = "o",
            s = 1,
            alpha = 0.25
            )

    # Plot the mean
    areas_averages = average_dicts_values_by_key(areas_average)
    ax.plot(
        *zip(*sorted(areas_averages.items())),
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
    Time of appearance frequency for either a single plate, or all plates on the lattice
    """
    _, ax = plt.subplots()
    colormap = cm.get_cmap("plasma")

    for plate_id, plate_item in plates_dict.items():
        if len(plates_dict) > 1:
            # Get a color from the colourmap
            cm_plate = colormap(0.2 + (0.65 - 0.2) * (plate_id / len(plates_dict)))
            plot_total = len(plates_dict)
        else:
            cm_plate = "Purple"
            plot_total = None

        if not len(plate_item) < 1:
            # Plot frequency for each time point
            time_of_appearance_frequency(ax, (plate_id, plate_item), time_points_elapsed, cm_plate, plot_total, bar = bar)

    lgd = ax.legend(loc = 'center right', fontsize = 8, bbox_to_anchor = (1.25, 0.5))
    save_params = {
        "format": "png",
        "bbox_extra_artists": (lgd,),
        "bbox_inches": "tight"
        }

    plt.ylim(ymin = 0)
    plt.title("Time of appearance")
    if bar:
        save_name = "time_of_appearance_bar.png"
    else:
        save_name = "time_of_appearance.png"
    plt.savefig(str(save_path.joinpath(save_name)), **save_params)

    plt.close()


def time_of_appearance_frequency(ax, plate_item, time_points_elapsed, plot_color, plot_total = None, bar = False):
    """
    Add a time of appearance frequency bar or line plot to an axis
    """
    plate_id, plate = plate_item

    time_points_dict = dict()
    for colony in plate.values():
        key = colony.timepoint_first.elapsed_minutes
        if key not in time_points_dict:
            time_points_dict[key] = 0
        time_points_dict[key] += 1

    # Normalise counts to frequency
    time_points_dict = {key: value / len(time_points_dict) for key, value in time_points_dict.items()}

    if not bar:
        ax.plot(
            *zip(*sorted(time_points_dict.items())),
            color = plot_color,
            label = f"Plate {plate_id}",
            alpha = 0.9
            )
    else:
        if plot_total is not None:
            width = plot_total + 1
            # Offset x positions so bars aren't obscured
            x = [x + ((plate_id - 1) * width) for x in sorted(time_points_dict.keys())]
        else:
            width = 14
            x = [x for x in sorted(time_points_dict.keys())]

        y = [time_points_dict[key] for key in sorted(time_points_dict.keys())]

        ax.bar(
            x,
            y,
            width = width,
            color = plot_color,
            label = f"Plate {plate_id}"
            )

    # Format x-axis labels as integer hours
    ax.set_xticklabels(axis_minutes_to_hours(ax.get_xticks()))
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("Frequency")


def plot_doubling_map(plates_dict, time_points_elapsed, save_path):
    """
    Heatmap of doubling time vs time of appearance
    """
    from numpy import histogram2d, zeros_like
    from numpy.ma import masked_where
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    _, ax = plt.subplots()
    x = [0]
    y = [0]

    for plate in plates_dict.values():
        for colony in plate.values():
            x.append(colony.timepoint_first.elapsed_minutes)
            y.append(colony.get_doubling_time_average(elapsed_minutes = True))

    # Normalise
    weights = zeros_like(x) + 1. / len(x)
    heatmap, xedges, yedges = histogram2d(x, y, bins = 50, weights = weights)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Mask out background values
    heatmap = masked_where(heatmap.T == 0, heatmap.T)

    img = plt.imshow(
        heatmap,
        cmap = "RdPu",
        extent = extent,
        origin = "lower"
        )

    plt.xlim(xmin = 0)
    plt.ylim(ymin = 0)
    plt.title("Appearance and doubling time distribution")

    # Add a divider so the colorbar will match the plot size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cb = plt.colorbar(img, cax)
    cb.set_label('Frequency')

    ax.set_xticklabels(axis_minutes_to_hours(ax.get_xticks()))
    ax.set_yticklabels(axis_minutes_to_hours(ax.get_yticks()))
    ax.set_xlabel("Time of appearance (hours)")
    ax.set_ylabel("Average doubling time (hours)")

    plt.tight_layout()
    plt.savefig(str(save_path.joinpath("appearance_doubling_distribution.png")), format = "png")

    plt.close()