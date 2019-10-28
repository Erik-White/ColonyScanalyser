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
    plotting,
    plots,
    colony
)
from .colony import Colony, timepoints_from_image, colonies_from_timepoints


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
    from pathlib import Path

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
        time_points.append(datetime.combine(datetime.strptime(date, "%Y%m%d"),datetime.strptime(times[i], "%H%M").time()))
    
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
    Finds all colonies on a plate and returns an array of co-ordinates

    If a co-ordinate is occupied by a colony, it contains that colonies labelled number

    :param plate_image: a black and white image as a numpy array
    :param mask: a black and white image as a numpy array
    :param plate_noise_mask: a black and white image as a numpy array
    :returns: a segmented and labelled image as a numpy array
    """
    from math import pi
    from scipy.ndimage.morphology import binary_fill_holes
    from skimage.morphology import remove_small_objects
    from skimage.measure import regionprops, label

    plate_image = imaging.remove_background_mask(plate_image, plate_mask)
    plate_noise_mask = imaging.remove_background_mask(plate_noise_mask, plate_mask)

    # Subtract an image of the first (i.e. empty) plate to remove static noise
    plate_image[plate_noise_mask] = 0

    # Fill any small gaps
    plate_image = binary_fill_holes(plate_image)

    # Remove background noise
    plate_image = remove_small_objects(plate_image, min_size = area_min)

    colonies = label(plate_image)

    # Remove colonies that are on the edge of the plate
    #versions <0.16 do not allow for a mask
    #colonies = clear_border(pl_th, buffer_size = 1, mask = plate_mask)

    # Exclude objects that are too eccentric
    rps = regionprops(colonies, coordinates = "rc")
    for rp in rps:
        # Eccentricity of zero is a perfect circle
        # Circularity of 1 is a perfect circle
        circularity = (4 * pi * rp.area) / (rp.perimeter * rp.perimeter)

        if rp.eccentricity > 0.6 or circularity < 0.85:
            colonies[colonies == rp.label] = 0

    return colonies


def image_file_to_timepoints(image_path, plate_coordinates, plate_images_mask, time_point, elapsed_minutes, plot_path = None):
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

    plate_timepoints = defaultdict(list)

    # Load image
    img = imread(str(image_path), as_gray = True)

    # Split image into individual plates
    plate_images = get_plate_images(img, plate_coordinates, edge_cut = 60)

    for j, plate_image in enumerate(plate_images):
        # Segment each image
        plate_images[j] = segment_image(plate_image, plate_image > 0, plate_images_mask[j], area_min = 8)
        # Create Timepoint objects for each plate
        plate_timepoints[j + 1].extend(timepoints_from_image(plate_images[j], time_point, elapsed_minutes))

        # Save segmented image plot, if required
        if plot_path is not None:
            plots.plot_plate_segmented(plate_image, plate_images[j], time_point, plot_path)

    return plate_timepoints
    

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
    parser.add_argument("--plate_lattice", type = int, nargs = 2, default = (3, 2),
                        metavar = ("ROW", "COL"),
                        help = "The row and column co-ordinate layout of plates. Example usage: --plate_lattice 3 3")
    parser.add_argument("--save_plots", type = int, default = 1,
                        help = "The detail level of plot images to store on disk")
    parser.add_argument("--use_saved", type = strtobool, default = True,
                        help = "Allow or prevent use of previously calculated data")
    parser.add_argument("-mp", "--multiprocessing", type = strtobool, default = True,
                        help = "Enables use of more CPU cores, faster but more resource intensive")

    args = parser.parse_args()
    BASE_PATH = args.path
    VERBOSE = args.verbose
    PLATE_SIZE = imaging.mm_to_pixels(args.plate_size - 5, dots_per_inch = args.dots_per_inch)
    PLATE_LATTICE = tuple(args.plate_lattice)
    SAVE_PLOTS = args.save_plots
    USE_SAVED = args.use_saved
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

    # Find images in working directory
    image_formats = ["tif", "tiff", "png"]
    image_files = file_access.get_files_by_type(BASE_PATH, image_formats)
    
    #Check if images have been loaded
    if len(image_files) > 0:
        if VERBOSE >= 1:
            print(f"{len(image_files)} images found")
    else:
        raise IOError(f"No images could be found in the supplied folder path. Images are expected in these formats: {image_formats}")

    # Get date and time information from filenames
    time_points = get_image_timestamps(image_files)
    time_points_elapsed = get_image_timestamps(image_files, elapsed_minutes = True)
    if len(time_points) != len(image_files) or len(time_points) != len(image_files):
        raise IOError("Unable to load timestamps from all image filenames. Please check that images have a filename with YYYYMMDD_HHMM timestamps")

    # Check if processed image data is already stored and can be loaded
    segmented_image_data_filename = "processed_data"
    if USE_SAVED:
        if VERBOSE >= 1:
            print("Attempting to load cached data")
        plate_colonies = file_access.load_file(
            BASE_PATH.joinpath("data", segmented_image_data_filename),
            file_access.CompressionMethod.LZMA,
            pickle = True
            )
        # Check that segmented image data has been loaded for all plates
        if VERBOSE >= 1 and plate_colonies is not None and len(plate_colonies) == utilities.coordinate_to_index_number(PLATE_LATTICE):
            print("Successfully loaded cached data")
        else:
            print("Unable to load cached data, starting image processing")
            plate_colonies = None
            
    # Process images to Timepoint data objects
    if not USE_SAVED or plate_colonies is None:
        plate_coordinates = None
        plate_images_mask = None
        plate_timepoints = defaultdict(list)

        if VERBOSE >= 1:
            print("Preprocessing images to locate plates")

        # Load the first image to get plate coordinate and mask
        with image_files[0] as image_file:
            # Load image
            img = imread(str(image_file), as_gray = True)

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
                if VERBOSE >= 3:
                    for k, center in enumerate(plate_coordinates, start = 1):
                        print(f"Plate {k} center: {center[0]}")

            # Split image into individual plates
            plate_images = get_plate_images(img, plate_coordinates, edge_cut = 60)

            # Use the first plate images as a noise mask
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
                # Allow args to be passed to callback function
                callback_function = partial(progress_update, progress = ((i + 1) / len(image_files)) * 100)

                # Create processes
                processes.append(pool.apply_async(
                    image_file_to_timepoints,
                    args = (image_file, plate_coordinates, plate_images_mask, time_points[i], time_points_elapsed[i]),
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
        plate_colonies = dict()
        for plate_id, plate in plate_timepoints.items():
            plate_colonies[plate_id] = {colony.id : colony for colony in colonies_from_timepoints(plate)}

            # Filter colonies to remove noise, background objects and merged colonies
            plate_colonies[plate_id] = dict(filter(lambda item:
                # Remove objects that do not have sufficient data points, usually just noise
                len(item[1].timepoints) > len(time_points) * 0.2 and
                # Remove object that do not show growth, these are not colonies
                item[1].growth_rate > 1 and
                # Colonies that appear with a large initial area are most likely merged colonies, not new colonies
                item[1].timepoint_first.area < 50,
                plate_colonies[plate_id].items()
                ))

            if VERBOSE >= 1:
                print(f"Colony data stored for {len(plate_colonies[plate_id])} colonies on plate {plate_id}")

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
        headers = [
            "Colony ID",
            "Time of appearance",
            "Time of appearance (elapsed minutes)",
            "Center point averaged (row, column)",
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
            plate.values(),
            headers,
            save_path.joinpath(f"plate{plate_id}_colonies")
            )

        # Save data for each colony on a plate
        headers = [
            "Colony ID",
            "Date/Time",
            "Elapsed time (minutes)",
            "Area (pixels)",
            "Center (row, column)",
            "Diameter (pixels)",
            "Perimeter (pixels)"
        ]
        colony_timepoints = list()
        for colony_id, colony in plate.items():
            for timepoint in colony.timepoints.values():
                # Unpack timepoint properties to a flat list
                colony_timepoints.append([colony_id, *timepoint])
                
        file_access.save_to_csv(
            colony_timepoints,
            headers,
            save_path.joinpath(f"plate{plate_id}_colony_timepoints")
            )
    
    if VERBOSE >= 1:
        print("Saving plots")
    # Plot colony growth curves and time of appearance for the plate
    if SAVE_PLOTS >= 2:
        for plate_id, plate in plate_colonies.items():
            row, col = utilities.index_number_to_coordinate(plate_id, PLATE_LATTICE)
            save_path = get_plate_directory(BASE_PATH.joinpath("plots"), row, col, create_dir = True)
            plate_item = {plate_id : plate}
            plots.plot_growth_curve(plate_item, time_points_elapsed, save_path)
            plots.plot_appearance_frequency(plate_item, time_points_elapsed, save_path)
            plots.plot_appearance_frequency(plate_item, time_points_elapsed, save_path, bar = True)

    # Plot colony growth curves for all plates
    if SAVE_PLOTS >= 1:
        save_path = file_access.create_subdirectory(BASE_PATH, "plots")
        plots.plot_growth_curve(plate_colonies, time_points_elapsed, save_path)
        plots.plot_appearance_frequency(plate_colonies, time_points_elapsed, save_path)
        plots.plot_appearance_frequency(plate_colonies, time_points_elapsed, save_path, bar = True)
        plots.plot_doubling_map(plate_colonies, time_points_elapsed, save_path)

    if VERBOSE >= 1:
        print(f"ColonyScanalyser analysis completed for: {BASE_PATH}")

    sys.exit()


if __name__ == "__main__":

    main()