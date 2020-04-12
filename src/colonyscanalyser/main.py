# System modules
import sys
import argparse
from typing import Union, Dict, List, Tuple
from pathlib import Path
from datetime import datetime
from distutils.util import strtobool
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

# Third party modules
from numpy import ndarray, diff

# Local modules
from colonyscanalyser import (
    utilities,
    file_access,
    imaging,
    plots
)
from .image_file import ImageFile, ImageFileCollection
from .plate import Plate, PlateCollection
from .colony import Colony, timepoints_from_image, colonies_from_timepoints, timepoints_from_image


def segment_image(
    plate_image: ndarray,
    plate_mask: ndarray = None,
    plate_noise_mask: ndarray = None,
    area_min: float = 1
) -> ndarray:
    """
    Attempts to separate and label all colonies on a plate

    :param plate_image: an image containing colonies
    :param plate_mask: a boolean image mask to remove from the original image
    :param plate_noise_mask: a black and white image as a numpy array
    :param area_min: the minimum area for a colony, in pixels
    :returns: a segmented and labelled image as a numpy array
    """
    from numpy import unique, isin
    from skimage.measure import regionprops, label
    from skimage.morphology import remove_small_objects, binary_erosion
    from skimage.segmentation import clear_border

    plate_image = imaging.remove_background_mask(plate_image, smoothing = 0.5)

    if plate_mask is not None:
        # Remove mask from image
        plate_image = plate_image & plate_mask
        # Remove objects touching the mask border
        plate_image = clear_border(plate_image, bgval = 0, mask = binary_erosion(plate_mask))
    else:
        # Remove objects touching the image border
        plate_image = clear_border(plate_image, buffer_size = 2, bgval = 0)

    plate_image = label(plate_image, connectivity = 2)

    # Remove background noise
    if len(unique(plate_image)) > 1:
        plate_image = remove_small_objects(plate_image, min_size = area_min)

    # Remove colonies that have grown on top of image artefacts or static objects
    if plate_noise_mask is not None:
        plate_noise_image = imaging.remove_background_mask(plate_noise_mask, smoothing = 0.5)
        if len(unique(plate_noise_mask)) > 1:
            noise_mask = remove_small_objects(plate_noise_image, min_size = area_min)
        # Remove all objects where there is an existing static object
        exclusion = unique(plate_image[noise_mask])
        exclusion_mask = isin(plate_image, exclusion[exclusion > 0])
        plate_image[exclusion_mask] = 0

    return plate_image


def image_file_to_timepoints(
    image_file: ndarray,
    plates: PlateCollection,
    plate_noise_masks: Dict[int, ndarray],
    plot_path: Path = None
) -> Dict[int, List[Colony.Timepoint]]:
    """
    Get Timepoint object data from a plate image

    Lists the results in a dict with the plate number as the key

    :param image_file: an ImageFile object
    :param plates: a PlateCollection of Plate instances
    :param plate_noise_masks: a dict of plate images to use as noise masks
    :param plot_path: a Path directory to save the segmented image plot
    :returns: a Dict of lists, each containing Timepoint objects
    """
    from collections import defaultdict
    from skimage.color import rgb2gray

    plate_timepoints = defaultdict(list)

    # Split image into individual plates
    plate_images = plates.slice_plate_image(image_file.image)

    for plate_id, plate_image in plate_images.items():
        plate_image_gray = rgb2gray(plate_image)
        # Segment each image
        plate_images[plate_id] = segment_image(plate_image_gray, plate_mask = plate_image_gray > 0, plate_noise_mask = plate_noise_masks[plate_id], area_min = 1.5)
        # Create Timepoint objects for each plate
        plate_timepoints[plate_id].extend(timepoints_from_image(plate_images[plate_id], image_file.timestamp_elapsed, image = plate_image))
        # Save segmented image plot, if required
        if plot_path is not None:
            save_path = file_access.create_subdirectory(plot_path, f"plate{plate_id}")
            plots.plot_plate_segmented(plate_image_gray, plate_images[plate_id], image_file.timestamp, save_path)

    return plate_timepoints


# flake8: noqa: C901
def main():
    parser = argparse.ArgumentParser(
        description = "An image analysis tool for measuring microorganism colony growth",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("path", type = str,
                        help = "Image files location", default = None)
    parser.add_argument("-dpi", "--dots_per_inch", type = int, default = 300,
                        help = "The image DPI (dots per inch) setting")
    parser.add_argument("-mp", "--multiprocessing", type = strtobool, default = True,
                        help = "Enables use of more CPU cores, faster but more resource intensive")
    parser.add_argument("-p", "--plots", type = int, default = 1,
                        help = "The detail level of plot images to store on disk")
    parser.add_argument("--plate_edge_cut", type = int, default = 5,
                        help = "The exclusion area from the plate edge, as a percentage of the plate diameter")
    parser.add_argument("--plate_labels", type = str, nargs = "*", default = list(),
                        help = "A list of labels to identify each plate. Plates are ordered from top left, in rows. Example usage: --plate_labels plate1 plate2")
    parser.add_argument("--plate_lattice", type = int, nargs = 2, default = (3, 2),
                        metavar = ("ROW", "COL"),
                        help = "The row and column co-ordinate layout of plates. Example usage: --plate_lattice 3 3")
    parser.add_argument("--plate_size", type = int, default = 90,
                        help = "The plate diameter, in millimetres")
    parser.add_argument("--use_cached_data", type = strtobool, default = False,
                        help = "Allow use of previously calculated data")
    parser.add_argument("-v", "--verbose", type = int, default = 1,
                        help = "Information output level")

    args = parser.parse_args()
    BASE_PATH = args.path
    PLOTS = args.plots
    PLATE_LABELS = {plate_id: label for plate_id, label in enumerate(args.plate_labels, start = 1)}
    PLATE_LATTICE = tuple(args.plate_lattice)
    PLATE_SIZE = int(imaging.mm_to_pixels(args.plate_size, dots_per_inch = args.dots_per_inch))
    PLATE_EDGE_CUT = int(round(PLATE_SIZE * (args.plate_edge_cut / 100)))
    USE_CACHED = args.use_cached_data
    VERBOSE = args.verbose
    POOL_MAX = 1
    if args.multiprocessing:
        POOL_MAX = cpu_count() - 1 if cpu_count() > 1 else 1

    if VERBOSE >= 1:
        print("Starting ColonyScanalyser analysis")
    if VERBOSE >= 2 and POOL_MAX > 1:
        print(f"Multiprocessing enabled, utilising {POOL_MAX} of {cpu_count()} processors")

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
    plates = None
    if USE_CACHED:
        if VERBOSE >= 1:
            print("Attempting to load cached data")
        plates = file_access.load_file(
            BASE_PATH.joinpath("data", segmented_image_data_filename),
            file_access.CompressionMethod.LZMA,
            pickle = True
        )
        # Check that segmented image data has been loaded for all plates
        # Also that data is not from an older format (< v0.4.0)
        if (
            VERBOSE >= 1 and plates is not None
            and plates.count == PlateCollection.coordinate_to_index(PLATE_LATTICE)
            and isinstance(plates.items[0], Plate)
        ):
            print("Successfully loaded cached data")
            image_files = None
        else:
            print("Unable to load cached data, starting image processing")
            plates = None

    if not USE_CACHED or plates is None:
        # Find images in working directory
        image_formats = ["tif", "tiff", "png"]
        image_paths = file_access.get_files_by_type(BASE_PATH, image_formats)

        # Store images as ImageFile objects
        # Timestamps are automatically read from filenames
        image_files = ImageFileCollection()
        for image_path in image_paths:
            image_files.add(
                file_path = image_path,
                timestamp = None,
                timestamp_initial = None,
                cache_image = False
            )

        # Check if images have been loaded and timestamps could be read
        if image_files.count > 0:
            if VERBOSE >= 1:
                print(f"{image_files.count} images found")
        else:
            raise IOError(f"No images could be found in the supplied folder path."
            " Images are expected in these formats: {image_formats}")
        if image_files.count != len(image_files.timestamps):
            raise IOError("Unable to load timestamps from all image filenames."
            " Please check that images have a filename with YYYYMMDD_HHMM timestamps")

        # Set intial timestamp
        image_files.timestamps_initial = image_files.timestamps[0]

        # Process images to Timepoint data objects
        plate_images_mask = None
        plate_timepoints = defaultdict(list)

        if VERBOSE >= 1:
            print("Preprocessing images to locate plates")

        # Load the first image to get plate coordinates and mask
        with image_files.items[0] as image_file:
            # Only find centers using first image. Assume plates do not move
            if plates is None:
                if VERBOSE >= 2:
                    print(f"Locating plate centres in image: {image_file.file_path}")

                # Create new Plate instances to store the information
                plates = PlateCollection.from_image(
                    shape = PLATE_LATTICE,
                    image = image_file.image_gray,
                    diameter = PLATE_SIZE,
                    search_radius = PLATE_SIZE // 20,
                    edge_cut = PLATE_EDGE_CUT,
                    labels = PLATE_LABELS
                )

                if not plates.count > 0:
                    print(f"Unable to locate plates in image: {image_file.file_path}")
                    print(f"Processing unable to continue")
                    sys.exit()
                
                if VERBOSE >= 3:
                    for plate in plates.items:
                        print(f"Plate {plate.id} center: {plate.center}")

            # Use the first plate image as a noise mask
            plate_noise_masks = plates.slice_plate_image(image_file.image_gray)

        if VERBOSE >= 1:
            print("Processing colony data from all images")

        # Thin wrapper to display a progress bar
        def progress_update(result, progress):
            utilities.progress_bar(progress, message = "Processing images")

        processes = list()
        with Pool(processes = POOL_MAX) as pool:
            for i, image_file in enumerate(image_files.items):
                # Allow args to be passed to callback function
                callback_function = partial(progress_update, progress = ((i + 1) / image_files.count) * 100)

                # Create processes
                processes.append(pool.apply_async(
                    image_file_to_timepoints,
                    args = (image_file, plates, plate_noise_masks),
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
        plate_noise_masks = None
        img = None

        if VERBOSE >= 1:
            print("Calculating colony properties")

        # Group Timepoints by centres and create Colony objects
        for plate_id, plate_timepoints in plate_timepoints.items():
            # If no objects are found
            if not len(plate_timepoints) > 0:
                break

            plate = plates.get_item(plate_id)
            plate.items = colonies_from_timepoints(plate_timepoints, distance_tolerance = 2)
            if VERBOSE >= 3:
                print(f"{plate.count} objects located on plate {plate.id}, before filtering")

            # Filter colonies to remove noise, background objects and merged colonies
            timestamp_diff_std = diff(image_files.timestamps_elapsed_seconds[1:]).std()
            timestamp_diff_std += 20
            plate.items = list(filter(lambda colony:
                # Remove objects that do not have sufficient data points
                len(colony.timepoints) > 5 and
                # No colonies should be visible at the start of the experiment
                colony.time_of_appearance.total_seconds() > 0 and
                # Remove objects with large gaps in the data
                diff([t.timestamp.total_seconds() for t in colony.timepoints[1:]]).std() < timestamp_diff_std and
                # Remove object that do not show growth, these are not colonies
                colony.timepoint_last.area > 4 * colony.timepoint_first.area and
                # Objects that appear with a large initial area are either merged colonies or noise
                colony.timepoint_first.area < 10,
                plate.items
            ))

            if VERBOSE >= 1:
                print(f"{plate.count} colonies identified on plate {plate.id}")

        if not any([plate.count for plate in plates.items]):
            if VERBOSE >= 1:
                print("Unable to locate any colonies in the images provided")
                print(f"ColonyScanalyser analysis completed for: {BASE_PATH}")
            sys.exit()

    # Store pickled data to allow quick re-use
    save_path = file_access.create_subdirectory(BASE_PATH, "data")
    save_path = save_path.joinpath(segmented_image_data_filename)
    save_status = file_access.save_file(save_path, plates, file_access.CompressionMethod.LZMA)
    if VERBOSE >= 1:
        if save_status:
            print(f"Cached data saved to {save_path}")
        else:
            print(f"An error occurred and cached data could not be written to disk at {save_path}")

    # Store colony data in CSV format
    if VERBOSE >= 1:
        print("Saving data to CSV")
        
    save_path = BASE_PATH.joinpath("data")
    for plate in plates.items:
        for colony in plate.items:
            test = colony.__iter__()
        # Save data for all colonies on one plate
        plate.colonies_to_csv(save_path)

        # Save data for each colony on a plate
        plate.colonies_timepoints_to_csv(save_path)

    # Save summarised data for all plates
    plates.plates_to_csv(save_path)

    # Only generate plots when working with original images
    # Can't guarantee that the original images and full list of time points
    # will be available when using cached data
    if image_files is not None:
        save_path = file_access.create_subdirectory(BASE_PATH, "plots")
        if PLOTS >= 1:
            if VERBOSE >= 1:
                print("Saving plots")
            # Summary plots for all plates
            plots.plot_growth_curve(plates.items, save_path)
            plots.plot_appearance_frequency(plates.items, save_path, timestamps = image_files.timestamps_elapsed)
            plots.plot_appearance_frequency(plates.items, save_path, timestamps = image_files.timestamps_elapsed, bar = True)
            plots.plot_doubling_map(plates.items, save_path)
            plots.plot_colony_map(image_files.items[-1].image, plates.items, save_path)

            for plate in plates.items:
                if VERBOSE >= 2:
                    print(f"Saving plots for plate #{plate.id}")
                save_path_plate = file_access.create_subdirectory(save_path, file_access.file_safe_name([f"plate{plate.id}", plate.name]))
                # Plot colony growth curves, ID map and time of appearance for each plate
                plots.plot_growth_curve([plate], save_path_plate)
                plots.plot_appearance_frequency([plate], save_path_plate, timestamps = image_files.timestamps_elapsed)
                plots.plot_appearance_frequency([plate], save_path_plate, timestamps = image_files.timestamps_elapsed, bar = True)

        if PLOTS >= 4:
            # Plot individual plate images as an animation
            if VERBOSE >= 1:
                print("Saving plate image animations. This may take several minutes")

            # Original size images
            plots.plot_plate_images_animation(
                plates,
                image_files,
                save_path,
                fps = 8,
                pool_max = POOL_MAX,
                image_size_maximum = (800, 800)
            )
            # Smaller images
            plots.plot_plate_images_animation(
                plates,
                image_files,
                save_path,
                fps = 8,
                pool_max = POOL_MAX,
                image_size = (250, 250),
                image_name = "plate_image_animation_small"
            )

    else:
        if VERBOSE >= 1:
            print("Unable to generate plots from cached data. Run analysis on original images to generate plot images")

    if VERBOSE >= 1:
        print(f"ColonyScanalyser analysis completed for: {BASE_PATH}")

    sys.exit()


if __name__ == "__main__":

    main()