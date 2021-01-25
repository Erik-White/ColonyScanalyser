from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes import Axes
from numpy import ndarray

from .plotting import rc_to_xy
from .plate import Plate, PlateCollection
from .image_file import ImageFile, ImageFileCollection


def plot_colony_map(plate_image: ndarray, plates: List[Plate], save_path: Path) -> Path:
    """
    Saves original plate image with overlaid plate and colony IDs

    :param plate_image: the final timepoint image of all plates
    :param plates: a PlateCollection of Plate instances
    :param save_path: the directory to save the plot image
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
    ax.axis("off")
    ax.imshow(plate_image)

    for plate in plates:
        center_y, center_x = plate.center

        # Colony coordinates are relative to individual plate images
        # Calculate a correction factor to allow plotting on the original image
        offset_y = center_y - plate.radius + plate.edge_cut
        offset_x = center_x - plate.radius + plate.edge_cut

        # Label plates
        ax.annotate(
            f"Plate #{plate.id}".upper(),
            (center_x, center_y - plate.radius - (plate.edge_cut * 1.4)),
            xycoords = "data",
            horizontalalignment = "center",
            verticalalignment = "center",
            fontsize = "40",
            backgroundcolor = "black",
            color = "white"
        )
        if len(plate.name) > 0:
            ax.annotate(
                plate.name,
                (center_x, center_y - plate.radius - (plate.edge_cut * 0.6)),
                xycoords = "data",
                horizontalalignment = "center",
                verticalalignment = "center",
                fontsize = "32",
                backgroundcolor = "black",
                color = "white"
            )

        # Mark the detected boundary of the plate
        plate_circle = plt.Circle(
            (center_x, center_y),
            radius = plate.radius,
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
            radius = plate.radius - plate.edge_cut,
            facecolor = "none",
            edgecolor = "white",
            linewidth = "1.5",
            linestyle = "--",
            label = "Colony detection area"
        )
        ax.add_artist(plate_circle_measured)

        # Mark colony centres and ID numbers
        for colony in plate.items:
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

    save_path = save_path.joinpath("plate_map.png")
    try:
        plt.savefig(str(save_path), format = "png")
    except Exception:
        save_path = None
    finally:
        plt.close()
        return save_path


def plot_plate_segmented(
    plate_image: ndarray,
    segmented_image: ndarray,
    date_time: datetime,
    save_path: Path
) -> Path:
    """
    Saves processed plate images and corresponding segmented data plots

    :param plate_image: a black and white image as a numpy array
    :param segmented_image: a segmented and labelled image as a numpy array
    :param date_time: a datetime object
    :param save_path: the directory to save the plot image
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


def plot_plate_images_animation(
    plates: PlateCollection,
    image_files: ImageFileCollection,
    save_path: Path,
    fps: int = 10,
    image_name: str = "plate_image_animation",
    pool_max: int = 1,
    **kwargs
) -> List[Path]:
    """
    Creates an animated gifs of individual plate images

    Utilizes multiprocessing to divide the list of images between processors

    :param plates: a PlateCollection instance
    :param image_files: an collection of ImageFile instances
    :param save_path: the directory to save the images
    :param fps: the framerate of the animated gif
    :param image_name: the file name for the saved gif images
    :param pool_max: the maximum number of processors to use for parallel processing
    :param kwargs: arguments to pass through to _image_file_to_plate_images
    :returns: a list of file path objects if the images were saved sucessfully
    """
    from multiprocessing import Pool
    from functools import partial
    from .file_access import create_subdirectory, file_safe_name

    # Divide up image files between processes and assemble results
    chunk_size = int(image_files.count // pool_max)
    func = partial(
        _image_file_to_plate_images,
        plate_collection = plates,
        **kwargs
    )
    with Pool(processes = pool_max) as pool:
        images = pool.map(
            func,
            image_files.items,
            chunksize = chunk_size
        )

    try:
        save_paths = list()
        for plate in plates.items:
            # Create directory for each plate if needed
            image_path = create_subdirectory(save_path, file_safe_name([f"plate{plate.id}", plate.name]))
            image_path = image_path.joinpath(image_name).with_suffix(".gif")
            save_paths.append(image_path)

            # Write image frames for each plate to disk
            images[0][plate.id].save(
                image_path,
                format = "GIF",
                save_all = True,
                append_images = [image[plate.id] for image in images[1:]],
                duration = int(1000 / fps),
                loop = 0
            )

    except Exception:
        save_paths = None
    finally:
        return save_paths


def plot_growth_curve(plates: List[Plate], save_path: Path) -> Path:
    """
    Growth curves for either a single plate, or all plates on the lattice

    :param plates: a list of Plate instances
    :param save_path: the directory to save the plot image
    :returns: a file path object if the plot was saved sucessfully
    """
    _, ax = plt.subplots()
    colormap = cm.get_cmap("plasma")
    growth_params = True

    for plate in plates:
        if len(plates) > 1:
            # Get a color from the colourmap
            cm_scatter = colormap(0.2 + (0.65 - 0.2) * (plate.id / len(plates)))
            cm_line = None
            growth_params = False
        else:
            cm_scatter = "Mediumpurple"
            cm_line = "Purple"

        # Add the growth curve plot for this plate
        growth_curve(ax, plate, cm_scatter, cm_line, growth_params = growth_params)

    ax.set_xlabel("Elapsed time (hours)")
    ax.set_xlim(0)
    ax.set_ylabel("log2[Colony area]")
    ax.set_ylim(0)

    lgd = ax.legend(loc = 'center right', fontsize = 8, bbox_to_anchor = (1.25, 0.5))
    save_params = {
        "format": "png",
        "bbox_extra_artists": (lgd,),
        "bbox_inches": "tight"
    }

    title = "Colony growth"
    if len(plates) == 1:
        title = _plate_title(title, plates[0])
    plt.title(title)

    save_path = save_path.joinpath("growth_curve.png")
    try:
        plt.savefig(str(save_path), **save_params)
    except Exception:
        save_path = None
    finally:
        plt.close()
        return save_path


def growth_curve(
    ax: Axes,
    plate: Plate,
    scatter_color: str,
    line_color: str = None,
    growth_params: bool = True
):
    """
    Add a growth curve scatter plot, with median, to an axis

    :param ax: a Matplotlib Axes object to add a plot to
    :param plate: a Plate instance
    :param scatter_color: a Colormap color
    :param line_color: a Colormap color for the median
    """
    from statistics import median
    from .utilities import savgol_filter

    if line_color is None:
        line_color = scatter_color

    for colony in plate.items:
        ax.scatter(
            # Matplotlib does not yet support timedeltas so we have to convert manually to float
            [td.total_seconds() / 3600 for td in sorted(colony.growth_curve.data.keys())],
            list(colony.growth_curve.data.values()),
            color = scatter_color,
            marker = "o",
            s = 1,
            alpha = 0.25
        )

    # Plot the smoothed median
    median = [median(val) for _, val in sorted(plate.growth_curve.data.items())]
    ax.plot(
        [td.total_seconds() / 3600 for td in sorted(plate.growth_curve.data.keys())],
        savgol_filter(median, 15, 2),
        color = line_color,
        label = "Smoothed median" if growth_params else f"Plate {plate.id}",
        linewidth = 2
    )

    if growth_params:
        # Plot lag, vmax and carrying capacity lines
        if plate.growth_curve.lag_time.total_seconds() > 0:
            line = ax.axvline(
                plate.growth_curve.lag_time.total_seconds() / 3600, color = "grey", linestyle = "dashed", alpha = 0.5
            )
            line.set_label("Lag time")

        if plate.growth_curve.carrying_capacity > 0:
            line = ax.axhline(plate.growth_curve.carrying_capacity, color = "blue", linestyle = "dashed", alpha = 0.5)
            line.set_label("Carrying\ncapacity")

        if plate.growth_curve.growth_rate > 0:
            y0, y1 = 0, plate.growth_curve.carrying_capacity
            x0 = plate.growth_curve.lag_time.total_seconds() / 3600
            x1 = ((y1 - y0) / (plate.growth_curve.growth_rate * 3600)) + x0
            ax.plot([x0, x1], [y0, y1], color = "red", linestyle = "dashed", alpha = 0.5, label = "Maximum\ngrowth rate")


def plot_appearance_frequency(  # noqa: C901
    plates: List[Plate],
    save_path: str,
    timestamps: List[timedelta] = None,
    bar = False
) -> Path:
    """
    Time of appearance frequency for either a single plate, or all plates on the lattice

    :param plates: a list of Plate instances
    :param save_path: the directory to save the plot image
    :param timestamps: a list of timedeltas used to rescale the plot
    :param bar: if a bar plot should be used instead of the default line plot
    :returns: a file path object if the plot was saved sucessfully
    """
    _, ax = plt.subplots()
    colormap = cm.get_cmap("plasma")
    figures = list()

    if len(plates) == 1 and not plates[0].count > 0:
        return

    for plate in plates:
        if not plate.count > 0:
            continue
        if len(plates) > 1:
            # Get a color from the colourmap
            cm_plate = colormap(0.2 + (0.65 - 0.2) * (plate.id / len(plates)))
        else:
            cm_plate = "Purple"

        # Plot frequency for each time point
        figures.extend(time_of_appearance_frequency(ax, plate, cm_plate, timestamps = timestamps, bar = bar))

    save_params = {
        "format": "png",
        "bbox_inches": "tight"
    }
    if len(plates) > 1:
        lgd = ax.legend(loc = 'center right', fontsize = 8, bbox_to_anchor = (1.25, 0.5))
        save_params["bbox_extra_artists"] = (lgd,)

    alpha = 1 / (0.25 * len(plates))
    if alpha > 1:
        alpha = 0.8
    for item in figures:
        item.set_alpha(alpha)

    plt.ylim(bottom = 0)
    if timestamps is not None:
        plt.xlim(min(timestamps).total_seconds() / 3600, max(timestamps).total_seconds() / 3600)
    title = "Time of appearance"
    if len(plates) == 1:
        title = _plate_title(title, plates[0])
    plt.title(title)

    if bar:
        save_name = "time_of_appearance_bar.png"
    else:
        save_name = "time_of_appearance.png"
    save_path = save_path.joinpath(save_name)

    try:
        plt.savefig(str(save_path), **save_params)
    except Exception:
        save_path = None
    finally:
        plt.close()
        return save_path


def time_of_appearance_frequency(
    ax: Axes,
    plate: Plate,
    plot_color: str,
    timestamps: List[timedelta] = None,
    bar: bool = False
) -> List:
    """
    Add a time of appearance frequency bar or line plot to an axis

    :param ax: a Matplotlib Axes object to add a plot to
    :param plate: a Plate instance
    :param plot_color: a Colormap color
    :param timestamps: a list of timedeltas used to rescale the plot
    :param bar: if a bar plot should be used instead of the default line plot
    :returns: either a list of matplotlib.lines.Line2D or matplotlib.patches.Rectangle objects
    """
    from collections import Counter

    timestamps = dict.fromkeys([timestamp.total_seconds() / 3600 for timestamp in timestamps], 0)
    appearance_counts = Counter(colony.time_of_appearance for colony in plate.items)

    # Normalise counts to frequency
    appearance_counts = {
        timestamp.total_seconds() / 3600: count / sum(appearance_counts.values())
        for timestamp, count in appearance_counts.items()
    }

    # Merge counts into default values dictionary
    appearance_counts = {**timestamps, **appearance_counts}

    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("Frequency")

    if bar:
        return ax.bar(
            *zip(*sorted(appearance_counts.items())),
            color = plot_color,
            label = f"Plate {plate.id}",
            alpha = 0.8
        )
    else:
        return ax.plot(
            *zip(*sorted(appearance_counts.items())),
            color = plot_color,
            label = f"Plate {plate.id}",
            alpha = 0.8
        )


def plot_doubling_map(plates: List[Plate], save_path: Path) -> Path:
    """
    Heatmap of doubling time vs time of appearance

    :param plates: a list of Plate instances
    :param save_path: the directory to save the plot image
    :returns: a file path object if the plot was saved sucessfully
    """
    from numpy import histogram2d, zeros_like
    from numpy.ma import masked_where
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    _, ax = plt.subplots()
    x = [0]
    y = [0]

    for plate in plates:
        if not plate.count > 0:
            return
        x.extend([colony.time_of_appearance.total_seconds() / 3600 for colony in plate.items])
        y.extend([colony.growth_curve.doubling_time.total_seconds() / 60 for colony in plate.items])

    # Normalise
    weights = zeros_like(x) + 1. / len(x)
    heatmap, xedges, yedges = histogram2d(x, y, bins = 50, weights = weights)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Mask out background values
    heatmap = masked_where(heatmap.T == 0, heatmap.T)

    img = plt.imshow(
        heatmap,
        cmap = "RdPu",
        aspect = "auto",
        extent = extent,
        origin = "lower"
    )

    plt.xlim(left = 0)
    plt.ylim(bottom = 0)
    plt.title("Appearance and doubling time distribution")

    # Add a divider so the colorbar will match the plot size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    cb = plt.colorbar(img, cax)
    cb.set_label('Frequency')

    ax.set_xlabel("Time of appearance (hours)")
    ax.set_ylabel("Doubling time (minutes)")

    plt.tight_layout()

    save_path = save_path.joinpath("appearance_doubling_distribution.png")
    try:
        plt.savefig(str(save_path), format = "png")
    except Exception:
        save_path = None
    finally:
        plt.close()
        return save_path


def _image_file_to_plate_images(
    image_file: ImageFile,
    plate_collection: PlateCollection = None,
    image_size: Tuple[int, int] = None,
    image_size_maximum: Tuple[int, int] = None,
    background_color = 0
) -> Dict[str, ndarray]:
    """
    Slice an image according to the plates in a PlateCollection

    Returns a set of PIL image objects

    :param image_file: the image containing a set of plates
    :param plate_collection: Plate objects shown in image_file
    :param image_size: the desired size for the individual plate images
    :param image_size_maximum: resize images if they are over a maximum threshold
    :param background_color: the color to replace the empty parts of the image
    :returns: a dictionary of plate images with the plate ID numbers as keys
    """
    from PIL import Image
    from skimage import img_as_ubyte
    from skimage.transform import resize

    # Slice the image into individual plate images
    sliced_images = plate_collection.slice_plate_image(image_file.image, background_color = background_color)
    # Create PIL Image objects and resize if required
    images = dict()
    for plate_id, image in sliced_images.items():
        if image_size is not None:
            # Resize to the desired size
            image = resize(image, image_size, preserve_range = True)
        # Check that the image is not over the maximum size
        if image_size_maximum is not None and image.shape > image_size_maximum:
            # Caluclate new image size while maintaining aspect ratio
            image_size = (image_size_maximum[0], image_size_maximum[0] * (image.shape[1] / image.shape[0]))
            # Resize if required
            image = resize(image, image_size, preserve_range = True)

        # skimage.transform.resize returns an array of type float64, which Image.fromarray can't handle
        images[plate_id] = Image.fromarray(img_as_ubyte(image), mode = "RGB")

    return images


def _plate_title(title: str, plate: Plate):
    """
    Construct a plot title containing the Plate information

    :param title: a plot title
    :param plate: a Plate instance
    :returns: a title containing the plate ID and label, if available
    """
    title_items = [title]

    if plate.name:
        title_items[0] += ","
        title_items.append(f"{plate.name}")
    title_items.append(f"(plate {plate.id})")

    return " ".join(title_items)