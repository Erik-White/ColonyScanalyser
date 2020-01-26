# System modules
import sys
import argparse
from pathlib import Path
from datetime import datetime
from distutils.util import strtobool
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

# Third party modules
from skimage.io import imread

# Local modules
from colonyscanalyser import (
    utilities,
    file_access,
    imaging,
    plots
)
from .plate import Plate
from .colony import Colony, timepoints_from_image, colonies_from_timepoints, timepoints_from_image


def get_plate_directory(parent_path, row, col, create_dir = True):
    """
    Determine the directory path for a specified plate

    Can create the directory if needed

    :param parent_path: a path object
    :param row: a lattice co-ordinate row
    :param col: a lattice co-ordinate column
    :param create_dir: specify if the directory should be created
    :returns: a path object for the specified plate
    """

    child_path = '_'.join(['row', str(row), 'col', str(col)])
    if create_dir:
        return file_access.create_subdirectory(parent_path, child_path)
    else:
        return parent_path.joinpath(child_path)


def get_image_timestamps(image_paths, elapsed_minutes = False):
    """
    Get timestamps from a list of images

    Assumes images have a file name with as timestamp
    Timestamps should be in YYYYMMDD_HHMM format

    :param images: a list of image file path objects
    :param elapsed_minutes: return timestamps as elapsed integer minutes
    :returns: a list of timestamps
    """
    time_points = list()

    # Get date and time information from filenames
    dates = [str(image.name[:-8].split("_")[-2]) for image in image_paths]
    times = [str(image.name[:-4].split("_")[-1]) for image in image_paths]
    
    # Convert string timestamps to Python datetime objects
    for i, date in enumerate(dates):
        time_points.append(datetime.combine(datetime.strptime(date, "%Y%m%d"), datetime.strptime(times[i], "%H%M").time()))

    if elapsed_minutes:
        # Store time points as elapsed minutes since start
        time_points_elapsed = list()
        for time_point in time_points:
            time_points_elapsed.append(int((time_point - time_points[0]).total_seconds() / 60))
        time_points = time_points_elapsed

    return time_points


def get_plate_images(image, plate_coordinates, edge_cut = 100):
    """
    Split image into lattice subimages and delete background

    :param img: a black and white image as a numpy array
    :param plate_coordinates: a list of centers and radii
    :param edge_cut: a radius, in pixels, to remove from the outer edge of the plate
    :returns: a list of plate images
    """
    plates = []

    for coordinate in plate_coordinates:
        center, radius = coordinate
        plates.append(imaging.cut_image_circle(image, center, radius - edge_cut))

    return plates


def segment_image(plate_image, plate_mask, plate_noise_mask, area_min = 5):
    """
    Attempts to find and label all colonies on a plate

    :param plate_image: a black and white image as a numpy array
    :param plate_mask: a black and white image as a numpy array
    :param plate_noise_mask: a black and white image as a numpy array
    :param area_min: the minimum area for a colony, in pixels
    :returns: a segmented and labelled image as a numpy array
    """
    from math import pi
    from scipy.ndimage.morphology import binary_fill_holes
    from skimage.morphology import remove_small_objects
    from skimage.measure import regionprops, label
    from skimage.segmentation import clear_border

    plate_image = imaging.remove_background_mask(plate_image, plate_mask)
    plate_noise_mask = imaging.remove_background_mask(plate_noise_mask, plate_mask)

    # Subtract an image of the first (i.e. empty) plate to remove static noise
    plate_image[plate_noise_mask] = 0

    # Fill any small gaps
    plate_image = binary_fill_holes(plate_image)

    # Remove background noise
    plate_image = remove_small_objects(plate_image, min_size = area_min)

    colonies = label(plate_image)

    # Remove colonies that are on the edge of the plate image
    colonies = clear_border(colonies, buffer_size = 1, mask = plate_mask)

    # Exclude objects that are too eccentric
    rps = regionprops(colonies)
    for rp in rps:
        # Eccentricity of zero is a perfect circle
        # Circularity of 1 is a perfect circle
        circularity = (4 * pi * rp.area) / (rp.perimeter * rp.perimeter)

        if rp.eccentricity > 0.5 or circularity < 0.65:
            colonies[colonies == rp.label] = 0

    return colonies


def image_file_to_timepoints(image_path, plate_coordinates, plate_images_mask, time_point, elapsed_minutes, edge_cut, plot_path = None):
    """
    Get Timepoint object data from a plate image

    Lists the results in a dict with the plate number as the key

    :param image_path: a Path object representing an image
    :param plate_coordinates: a list of (row, column) tuple plate centres
    :param plate_images_mask: a list of plate images to use as noise masks
    :param time_point: a Datetime object
    :param elapsed_minutes: the number of integer minutes since starting
    :param plot_path: a Path directory to save the segmented image plot
    :returns: a Dict of lists, each containing Timepoint objects
    """
    from collections import defaultdict
    from skimage.color import rgb2gray

    plate_timepoints = defaultdict(list)

    # Load image
    img = imread(str(image_path), plugin = "pil", as_gray = False)

    # Split image into individual plates
    plate_images = get_plate_images(img, plate_coordinates, edge_cut = edge_cut)

    for j, plate_image in enumerate(plate_images):
        plate_image_gray = rgb2gray(plate_image)
        # Segment each image
        plate_images[j] = segment_image(plate_image_gray, plate_image_gray > 0, plate_images_mask[j], area_min = 8)
        # Create Timepoint objects for each plate
        plate_timepoints[j + 1].extend(timepoints_from_image(plate_images[j], time_point, elapsed_minutes, image = plate_image))
        # Save segmented image plot, if required
        if plot_path is not None:
            plots.plot_plate_segmented(plate_image_gray, plate_images[j], time_point, plot_path)

    return plate_timepoints


# flake8: noqa: C901
def main():
    parser = argparse.ArgumentParser(
        description = "An image analysis tool for measuring microorganism colony growth",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("path", type = str,
                        help = "Image files location", default = None)
    parser.add_argument("-v", "--verbose", type = int, default = 1,
                        help = "Information output level")
    parser.add_argument("-dpi", "--dots_per_inch", type = int, default = 2540,
                        help = "The image DPI (dots per inch) setting")
    parser.add_argument("--plate_size", type = int, default = 100,
                        help = "The plate diameter, in millimetres")
    parser.add_argument("--plate_edge_cut", type = int, default = 60,
                        help = "The radius from the plate edge to remove, in pixels")
    parser.add_argument("--plate_lattice", type = int, nargs = 2, default = (3, 2),
                        metavar = ("ROW", "COL"),
                        help = "The row and column co-ordinate layout of plates. Example usage: --plate_lattice 3 3")
    parser.add_argument("--plate_labels", type = str, nargs = "*", default = list(),
                        help = "A list of labels to identify each plate. Plates are ordered from top left, in rows. Example usage: --plate_labels plate1 plate2")
    parser.add_argument("--save_plots", type = int, default = 1,
                        help = "The detail level of plot images to store on disk")
    parser.add_argument("--use_cached_data", type = strtobool, default = False,
                        help = "Allow use of previously calculated data")
    parser.add_argument("-mp", "--multiprocessing", type = strtobool, default = True,
                        help = "Enables use of more CPU cores, faster but more resource intensive")

    args = parser.parse_args()
    BASE_PATH = args.path
    VERBOSE = args.verbose
    PLATE_SIZE = imaging.mm_to_pixels(args.plate_size, dots_per_inch = args.dots_per_inch)
    PLATE_LATTICE = tuple(args.plate_lattice)
    PLATE_LABELS = {plate_id: label for plate_id, label in enumerate(args.plate_labels, start = 1)}
    PLATE_EDGE_CUT = args.plate_edge_cut
    SAVE_PLOTS = args.save_plots
    USE_CACHED = args.use_cached_data
    POOL_MAX = 1
    if args.multiprocessing:
        POOL_MAX = cpu_count()

    if VERBOSE >= 1:
        print("Starting ColonyScanalyser analysis")

    # Resolve working directory
    if BASE_PATH is None:
        raise ValueError("A path to a working directory must be supplied")
    else:
        BASE_PATH = Path(args.path).resolve()
    if not BASE_PATH.exists():
        raise EnvironmentError(f"The supplied folder path could not be found: {BASE_PATH}")
    if VERBOSE >= 1:
        print(f"Working directory: {BASE_PATH}")

    # Check if processed image data is already stored and can be loaded
    segmented_image_data_filename = "cached_data"
    if USE_CACHED:
        if VERBOSE >= 1:
            print("Attempting to load cached data")
        plate_colonies = file_access.load_file(
            BASE_PATH.joinpath("data", segmented_image_data_filename),
            file_access.CompressionMethod.LZMA,
            pickle = True
        )
        # Check that segmented image data has been loaded for all plates
        # Also that data is not from an older format (< v0.4.0)
        if (
            VERBOSE >= 1 and plate_colonies is not None
            and len(plate_colonies) == utilities.coordinate_to_index_number(PLATE_LATTICE)
            and isinstance(plate_colonies[1], Plate)
        ):
            print("Successfully loaded cached data")
            image_files = None
        else:
            print("Unable to load cached data, starting image processing")
            plate_colonies = None

    if not USE_CACHED or plate_colonies is None:
        # Find images in working directory
        image_formats = ["tif", "tiff", "png"]
        image_files = file_access.get_files_by_type(BASE_PATH, image_formats)

        # Check if images have been loaded
        if len(image_files) > 0:
            if VERBOSE >= 1:
                print(f"{len(image_files)} images found")
        else:
            raise IOError(f"No images could be found in the supplied folder path."
            " Images are expected in these formats: {image_formats}")

        # Get date and time information from filenames
        time_points = get_image_timestamps(image_files)
        time_points_elapsed = get_image_timestamps(image_files, elapsed_minutes = True)
        if len(time_points) != len(image_files) or len(time_points) != len(image_files):
            raise IOError("Unable to load timestamps from all image filenames."
            " Please check that images have a filename with YYYYMMDD_HHMM timestamps")

        # Process images to Timepoint data objects
        plate_coordinates = None
        plate_images_mask = None
        plate_timepoints = defaultdict(list)

        if VERBOSE >= 1:
            print("Preprocessing images to locate plates")

        # Load the first image to get plate coordinate and mask
        with image_files[0] as image_file:
            # Load image
            img = imread(str(image_file), plugin = "pil", as_gray = True)

            # Only find centers using first image. Assume plates do not move
            if plate_coordinates is None:
                if VERBOSE >= 2:
                    print(f"Locating plate centres in image: {image_file}")
                plate_coordinates = imaging.get_image_circles(
                    img,
                    int(PLATE_SIZE / 2),
                    circle_count = utilities.coordinate_to_index_number(PLATE_LATTICE),
                    search_radius = 50
                )
                # Create new Plate instances to store the information
                plate_colonies = dict()
                for plate_id, coord in enumerate(plate_coordinates, start = 1):
                    center, radius = coord

                    plate = Plate(plate_id, radius * 2)
                    plate.center = center
                    plate.edge_cut = PLATE_EDGE_CUT
                    if plate.id in PLATE_LABELS:
                        plate.name = PLATE_LABELS[plate.id]

                    plate_colonies[plate.id] = plate

                if not len(plate_colonies) > 0:
                    print(f"Unable to locate plates in image: {image_file}")
                    print(f"Processing unable to continue")
                    sys.exit()
                
                if VERBOSE >= 3:
                    for plate in plate_colonies.values():
                        print(f"Plate {plate.id} center: {plate.center}")

            # Split image into individual plates
            plate_images = get_plate_images(img, plate_coordinates, edge_cut = PLATE_EDGE_CUT)

            # Use the first plate image as a noise mask
            if plate_images_mask is None:
                plate_images_mask = plate_images

        if VERBOSE >= 1:
            print("Processing colony data from all images")

        # Thin wrapper to display a progress bar
        def progress_update(result, progress):
            utilities.progress_bar(progress, message = "Processing images")

        processes = list()
        with Pool(processes = POOL_MAX) as pool:
            for i, image_file in enumerate(image_files):
                # Load image
                img = imread(str(image_file), plugin = "pil", as_gray = True)
                        
                # Allow args to be passed to callback function
                callback_function = partial(progress_update, progress = ((i + 1) / len(image_files)) * 100)

                # Create processes
                processes.append(pool.apply_async(
                    image_file_to_timepoints,
                    args = (image_file, plate_coordinates, plate_images_mask, time_points[i], time_points_elapsed[i], PLATE_EDGE_CUT),
                    kwds = {"plot_path" : None},
                    callback = callback_function
                ))

            # Consolidate the results to a single dict
            for process in processes:
                result = process.get()
                for plate_id, timepoints in result.items():
                    plate_timepoints[plate_id].extend(timepoints)

        # Clear objects to free up memory
        processes = None
        plate_images = None
        plate_images_mask = None
        img = None

        if VERBOSE >= 1:
            print("Calculating colony properties")

        # Group Timepoints by centres and create Colony objects
        for plate_id, plate in plate_timepoints.items():
            plate_colonies[plate_id].colonies = colonies_from_timepoints(plate, distance_tolerance = 8)
            if VERBOSE >= 3:
                print(f"{plate_colonies[plate_id].colony_count} colonies located on plate {plate_id}, before filtering")

            # Filter colonies to remove noise, background objects and merged colonies
            plate_colonies[plate_id].colonies = list(filter(lambda item:
                # Remove objects that do not have sufficient data points, usually just noise
                len(item.timepoints) > len(time_points) * 0.2 and
                # Remove object that do not show growth, these are not colonies
                item.growth_rate > 1 and
                # Colonies that appear with a large initial area are most likely merged colonies, not new colonies
                item.timepoint_first.area < 50,
                plate_colonies[plate_id].colonies
            ))

            if VERBOSE >= 1:
                print(f"Colony data stored for {plate_colonies[plate_id].colony_count} colonies on plate {plate_id}")

        if not any([len(plate.colonies) for plate in plate_colonies.values()]):
            if VERBOSE >= 1:
                print("Unable to locate any colonies in the images provided")
                print(f"ColonyScanalyser analysis completed for: {BASE_PATH}")
            sys.exit()

    # Store pickled data to allow quick re-use
    save_path = file_access.create_subdirectory(BASE_PATH, "data")
    save_path = save_path.joinpath(segmented_image_data_filename)
    save_status = file_access.save_file(save_path, plate_colonies, file_access.CompressionMethod.LZMA)
    if VERBOSE >= 1:
        if save_status:
            print(f"Cached data saved to {save_path}")
        else:
            print(f"An error occurred and cached data could not be written to disk at {save_path}")

    # Store colony data in CSV format
    if VERBOSE >= 1:
        print("Saving data to CSV")
        
    save_path = BASE_PATH.joinpath("data")
    for plate_id, plate in plate_colonies.items():
        plate_name = ""
        if len(plate.name) > 0:
            plate_name = "_" + plate.name.replace(" ", "_")

        headers = [
            "Colony ID",
            "Time of appearance",
            "Time of appearance (elapsed minutes)",
            "Center point averaged (row, column)",
            "Colour averaged name",
            "Colour averaged (R,G,B)",
            "Growth rate average",
            "Growth rate",
            "Doubling time average (minutes)",
            "Doubling times (minutes)",
            "First detection (elapsed minutes)",
            "First center point (row, column)",
            "First area (pixels)",
            "First diameter (pixels)",
            "Final detection (elapsed minutes)",
            "Final center point (row, column)",
            "Final area (pixels)",
            "Final diameter (pixels)"
        ]

        # Save data for all colonies on one plate
        file_access.save_to_csv(
            plate.colonies,
            headers,
            save_path.joinpath(f"plate{str(plate_id) + plate_name}_colonies")
            )

        # Save data for each colony on a plate
        headers = [
            "Colony ID",
            "Date/Time",
            "Elapsed time (minutes)",
            "Area (pixels)",
            "Center (row, column)",
            "Diameter (pixels)",
            "Perimeter (pixels)",
            "Color average (R,G,B)"
        ]
        colony_timepoints = list()
        for colony in plate.colonies:
            for timepoint in colony.timepoints.values():
                # Unpack timepoint properties to a flat list
                colony_timepoints.append([colony.id, *timepoint])

        file_access.save_to_csv(
            colony_timepoints,
            headers,
            save_path.joinpath(f"plate{str(plate_id) + plate_name}_colony_timepoints")
        )

    # Only generate plots when working with original images
    # Can't guarantee that the original images and full list of time points
    # will be available when using cached data
    if image_files is not None:
        # Plots for all plates
        if SAVE_PLOTS >= 1:
            if VERBOSE >= 1:
                print("Saving plots")
            save_path = file_access.create_subdirectory(BASE_PATH, "plots")
            plots.plot_growth_curve(plate_colonies, time_points_elapsed, save_path)
            plots.plot_appearance_frequency(plate_colonies, time_points_elapsed, save_path)
            plots.plot_appearance_frequency(plate_colonies, time_points_elapsed, save_path, bar = True)
            plots.plot_doubling_map(plate_colonies, time_points_elapsed, save_path)
            plots.plot_colony_map(imread(image_files[-1], plugin = "pil", as_gray = False), plate_colonies, save_path)

        # Plot colony growth curves, ID map and time of appearance for each plate
        if SAVE_PLOTS >= 2:
            for plate_id, plate in plate_colonies.items():
                row, col = utilities.index_number_to_coordinate(plate_id, PLATE_LATTICE)
                save_path_plate = get_plate_directory(save_path, row, col, create_dir = True)
                plate_item = {plate_id : plate}
                plots.plot_growth_curve(plate_item, time_points_elapsed, save_path_plate)
                plots.plot_appearance_frequency(plate_item, time_points_elapsed, save_path_plate)
                plots.plot_appearance_frequency(plate_item, time_points_elapsed, save_path_plate, bar = True)
    else:
        if VERBOSE >= 1:
            print("Unable to generate plots from cached data. Run analysis on original images to generate plot images")

    if VERBOSE >= 1:
        print(f"ColonyScanalyser analysis completed for: {BASE_PATH}")

    sys.exit()


if __name__ == "__main__":

    main()