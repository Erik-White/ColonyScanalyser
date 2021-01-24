# System modules
import sys
import argparse
from typing import Optional, Union, Dict, List, Tuple
from pathlib import Path
from datetime import datetime
from distutils.util import strtobool
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
try:
    from importlib import metadata
except ImportError:
    # Running on pre-3.8 Python, use importlib-metadata package
    import importlib_metadata as metadata

# Third party modules
from numpy import ndarray, diff

# Local modules
from colonyscanalyser import (
    config,
    utilities,
    file_access,
    imaging,
    plots
)
from .image_file import ImageFile, ImageFileCollection
from .plate import Plate, PlateCollection
from .colony import Colony, timepoints_from_image, colonies_filtered, colonies_from_timepoints, timepoints_from_image
from .align.strategy import AlignStrategy, apply_align_transform, calculate_transformation_strategy
from .align.transform import AlignTransform


def argparse_init(*args, **kwargs) -> argparse.ArgumentParser:
    """
    Initialise an ArgumentParser instance with the standard package arguments

    :param args: positional arguments to pass to the ArgumentParser initialiser
    :param kwargs: keyword arguments to pass to the ArgumentParser initialiser
    :returns: an ArgumentParser instance with the standard package arguments
    """
    parser = argparse.ArgumentParser(*args, **kwargs)

    # Mutually exclusive options
    output = parser.add_mutually_exclusive_group()
    
    parser.add_argument("path", type = str,
                        help = "Image files location", default = None)
    parser.add_argument("-a", "--animation", action = "store_true",
                        help = "Output animated plots and videos")
    parser.add_argument("-d", "--dots-per-inch", type = int, default = config.DOTS_PER_INCH, metavar = "N",
                        help = "The image DPI (dots per inch) setting")
    parser.add_argument("--image-align", nargs = "?", default = AlignStrategy.quick.name, const = AlignStrategy.quick.name, choices = [strategy.name for strategy in AlignStrategy],
                        help = "The strategy used for aligning images for analysis")
    parser.add_argument("--image-align-tolerance", type = float, default = config.ALIGNMENT_TOLERANCE,
                        help = "The tolerance value allowed when aligning images. 0 means the images must match exactly", metavar = "N")
    parser.add_argument("--image-formats", default = config.SUPPORTED_FORMATS, action = "version", version = str(config.SUPPORTED_FORMATS),
                        help = "The supported image formats")
    parser.add_argument("--no-plots", action = "store_true", help = "Prevent output of plot images to disk")
    parser.add_argument("--plate-edge-cut", type = int, default = config.PLATE_EDGE_CUT,
                        help = "The exclusion area from the plate edge, as a percentage of the plate diameter", metavar = "N")
    parser.add_argument("--plate-labels", type = str, nargs = "*", default = list(), metavar = "LABEL",
                        help = "A list of labels to identify each plate. Plates are ordered from top left, in rows. Example usage: --plate_labels plate1 plate2")
    parser.add_argument("--plate-lattice", type = int, nargs = 2, default = config.PLATE_LATTICE, metavar = ("ROW", "COL"),
                        help = "The row and column co-ordinate layout of plates. Example usage: --plate_lattice 3 3")
    parser.add_argument("--plate-size", type = int, default = config.PLATE_SIZE, help = "The plate diameter, in millimetres", metavar = "N")
    output.add_argument("-s", "--silent", action = "store_true", help = "Silence all output to console")
    parser.add_argument("--single-process", action = "store_true", help = "Use only a single CPU core, slower but less resource intensive")
    parser.add_argument("-u", "--use-cached-data", action = "store_true", help = "Allow use of previously calculated data")
    output.add_argument("-v", "--verbose", action = "store_true", help = "Output extra information to console")
    parser.add_argument("--version", action = "version", version = f"ColonyScanlayser {metadata.version('colonyscanalyser')}",
                        help = "The package version number")

    return parser


def plates_colonies_from_timepoints(
    plates: PlateCollection,
    timepoints: Dict[int, List[Colony.Timepoint]],
    timepoints_distance: float = 1,
    timestamp_diff_std: float = 10,
    pool_size = 1
) -> Plate:
    """
    Group a list of Timepoints to Colony objects, and populate in a Plate instance

    Colonies are filtered via the criteria of the Colony.colonies_filtered function

    :param plates: a PlateCollection instance associated with the Timepoints
    :param timepoints: a dict of lists of Timepoint instances, with keys corresponding to Plate.id numbers
    :param timepoints_distance: the maximum distance allowed for Colony grouping
    :param timestamp_diff_std: the maximum allowed deviation in timestamps (i.e. likelihood of missing data)
    :param pool_size: the number of logical processors available for multiprocessing
    :returns: the collection with each plate instance populated with a collection of Colony instances
    """
    # Assemble data to a single iterable for starmap
    timepoints_iter = [
        (plates[plate_id], timepoints, timepoints_distance, timestamp_diff_std)
        for plate_id, timepoints in timepoints.items()
    ]

    # Process and filter Timepoints to Colony objects in parallel
    with Pool(processes = pool_size) as pool:
        plates.items = pool.starmap(
            func = _plate_colonies_from_timepoints_filtered,
            iterable = timepoints_iter
        )

    return plates


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
    image_file: ImageFile,
    plates: PlateCollection,
    plate_noise_masks: Dict[int, ndarray]
) -> Dict[int, List[Colony.Timepoint]]:
    """
    Get Timepoint object data from a plate image

    :param image_file: an ImageFile object
    :param plates: a PlateCollection of Plate instances
    :param plate_noise_masks: a dict of plate images to use as noise masks
    :returns: a Dict of lists containing Timepoints, with the plate number as keys
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

    return plate_timepoints


def _plate_colonies_from_timepoints_filtered(
    plate: Plate,
    timepoints: List[Colony.Timepoint],
    timepoints_distance: float = 1,
    timestamp_diff_std: float = 10
) -> Plate:
    """
    Group a list of Timepoints to Colony objects, and filter to return only valid colonies

    Mainly a helper function for plates_colonies_from_timepoints as multiprocessing cant use local functions

    :param plate: a Plate instance associated with the Timepoints
    :param timepoints: a list of Timepoint instances
    :param timepoints_distance: the maximum distance allowed for Colony grouping
    :param timestamp_diff_std: the maximum allowed deviation in timestamps (i.e. likelihood of missing data)
    :returns: the plate instance with a collection of Colony instances
    """
    if len(timepoints) > 0:
        # Group Timepoints by Euclidean distance
        plate.items = colonies_from_timepoints(timepoints, distance_tolerance = timepoints_distance)

        # Filter colonies to remove noise, background objects and merged colonies
        plate.items = colonies_filtered(plate.items, timestamp_diff_std)

    return plate


# flake8: noqa: C901
def main():
    parser = argparse_init(
        description = "An image analysis tool for measuring microorganism colony growth",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        usage = "%(prog)s '/image/file/path/' [OPTIONS]"
        )

    # Retrieve and parse arguments
    args = parser.parse_args()
    BASE_PATH = args.path
    ANIMATION = args.animation
    IMAGE_ALIGN_STRATEGY = AlignStrategy[args.image_align]
    IMAGE_ALIGN_TOLERANCE = args.image_align_tolerance
    IMAGE_FORMATS = args.image_formats
    PLOTS = not args.no_plots
    PLATE_LABELS = {plate_id: label for plate_id, label in enumerate(args.plate_labels, start = 1)}
    PLATE_LATTICE = tuple(args.plate_lattice)
    PLATE_SIZE = int(imaging.mm_to_pixels(args.plate_size, dots_per_inch = args.dots_per_inch))
    PLATE_EDGE_CUT = int(round(PLATE_SIZE * (args.plate_edge_cut / 100)))
    SILENT = args.silent
    USE_CACHED = args.use_cached_data
    VERBOSE = args.verbose
    POOL_MAX = 1
    if not args.single_process:
        POOL_MAX = cpu_count() - 1 if cpu_count() > 1 else 1
        
    if not SILENT:
        print("Starting ColonyScanalyser analysis")
    if VERBOSE and POOL_MAX > 1:
        print(f"Multiprocessing enabled, utilising {POOL_MAX} of {cpu_count()} processors")

    # Resolve working directory
    if BASE_PATH is None:
        raise ValueError("A path to a working directory must be supplied")
    else:
        BASE_PATH = Path(args.path).resolve()
    if not BASE_PATH.exists():
        raise EnvironmentError(f"The supplied folder path could not be found: {BASE_PATH}")
    if not SILENT:
        print(f"Working directory: {BASE_PATH}")

    # Check if processed image data is already stored and can be loaded
    plates = None
    if USE_CACHED:
        if not SILENT:
            print("Attempting to load cached data")
        plates = file_access.load_file(
            BASE_PATH.joinpath(config.DATA_DIR, config.CACHED_DATA_FILE_NAME),
            file_access.CompressionMethod.LZMA,
            pickle = True
        )
        # Check that segmented image data has been loaded for all plates
        # Also that data is not from an older format (< v0.4.0)
        if (
            VERBOSE and plates is not None
            and plates.count == PlateCollection.coordinate_to_index(PLATE_LATTICE)
            and isinstance(plates.items[0], Plate)
        ):
            print("Successfully loaded cached data")
            image_files = None
        else:
            print("Unable to load cached data, starting image processing")
            plates = None

    if not USE_CACHED or plates is None:
        # Find images in working directory. Raises IOError if images not loaded correctly
        image_files = ImageFileCollection.from_path(BASE_PATH, IMAGE_FORMATS, cache_images = False)
        if not SILENT:
            print(f"{image_files.count} images found")

        # Verify image alignment
        if IMAGE_ALIGN_STRATEGY != AlignStrategy.none:
            if not SILENT:
                print(f"Verifying image alignment with '{IMAGE_ALIGN_STRATEGY.name}' strategy. This process will take some time")
            
            # Initialise the model and determine which images need alignment
            align_model, image_files_align = calculate_transformation_strategy(
                image_files.items,
                IMAGE_ALIGN_STRATEGY,
                tolerance = IMAGE_ALIGN_TOLERANCE
            )
            
            # Apply image alignment according to selected strategy
            if len(image_files_align) > 0:
                if not SILENT:
                    print(f"{len(image_files_align)} of {image_files.count} images require alignment")

                with Pool(processes = POOL_MAX) as pool:
                    results = list()
                    job = pool.imap_unordered(
                        func = partial(apply_align_transform, align_model = align_model),
                        iterable = image_files_align,
                        chunksize = 2
                    )
                    # Store results and update progress bar
                    for i, result in enumerate(job, start = 1):
                        results.append(result)
                        if not SILENT:
                            utilities.progress_bar((i / len(image_files_align)) * 100, message = "Correcting image alignment")

                    image_files.update(results)

        # Process images to Timepoint data objects
        plate_images_mask = None
        plate_timepoints = defaultdict(list)

        if not SILENT:
            print("Preprocessing images to locate plates")

        # Load the first image to get plate coordinates and mask
        with image_files.items[0] as image_file:
            # Only find centers using first image. Assume plates do not move
            if plates is None:
                if VERBOSE:
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
                    if not SILENT:
                        print(f"Unable to locate plates in image: {image_file.file_path}")
                        print(f"Processing unable to continue")
                    sys.exit()
                
                if VERBOSE:
                    for plate in plates.items:
                        print(f"Plate {plate.id} center: {plate.center}")

            # Use the first plate image as a noise mask
            plate_noise_masks = plates.slice_plate_image(image_file.image_gray)

        if not SILENT:
            print("Processing colony data from all images")

        # Process images to Timepoints
        with Pool(processes = POOL_MAX) as pool:
            results = list()
            job = pool.imap(
                func = partial(image_file_to_timepoints, plates = plates, plate_noise_masks = plate_noise_masks),
                iterable = image_files.items,
                chunksize = 2
            )
            # Store results and update progress bar
            for i, result in enumerate(job, start = 1):
                results.append(result)
                if not SILENT:
                    utilities.progress_bar((i / image_files.count) * 100, message = "Processing images")
            plate_timepoints = utilities.dicts_merge(list(results))

        if not SILENT:
            print("Calculating colony properties")

        # Calculate deviation in timestamps (i.e. likelihood of missing data)
        timestamp_diff_std = diff(image_files.timestamps_elapsed_seconds[1:]).std()
        timestamp_diff_std += config.COLONY_TIMESTAMP_DIFF_MAX

        # Group and consolidate Timepoints into Colony instances
        plates = plates_colonies_from_timepoints(plates, plate_timepoints, config.COLONY_DISTANCE_MAX, timestamp_diff_std, POOL_MAX)

        if not any([plate.count for plate in plates.items]):
            if not SILENT:
                print("Unable to locate any colonies in the images provided")
                print(f"ColonyScanalyser analysis completed for: {BASE_PATH}")
            sys.exit()
        elif not SILENT:
            for plate in plates.items:
                print(f"{plate.count} colonies identified on plate {plate.id}")

    # Store pickled data to allow quick re-use
    save_path = file_access.create_subdirectory(BASE_PATH, config.DATA_DIR)
    save_path = save_path.joinpath(config.CACHED_DATA_FILE_NAME)
    save_status = file_access.save_file(save_path, plates, file_access.CompressionMethod.LZMA)
    if not SILENT:
        if save_status:
            print(f"Cached data saved to {save_path}")
        else:
            print(f"An error occurred and cached data could not be written to disk at {save_path}")

    # Store colony data in CSV format
    if not SILENT:
        print("Saving data to CSV")
        
    save_path = BASE_PATH.joinpath(config.DATA_DIR)
    for plate in plates.items:
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
        if PLOTS or ANIMATION:
            save_path = file_access.create_subdirectory(BASE_PATH, config.PLOTS_DIR)
        if PLOTS:
            if not SILENT:
                print("Saving plots")
            # Summary plots for all plates
            plots.plot_growth_curve(plates.items, save_path)
            plots.plot_appearance_frequency(plates.items, save_path, timestamps = image_files.timestamps_elapsed)
            plots.plot_appearance_frequency(plates.items, save_path, timestamps = image_files.timestamps_elapsed, bar = True)
            plots.plot_doubling_map(plates.items, save_path)
            plots.plot_colony_map(image_files.items[-1].image, plates.items, save_path)

            for plate in plates.items:
                if VERBOSE:
                    print(f"Saving plots for plate #{plate.id}")
                save_path_plate = file_access.create_subdirectory(save_path, file_access.file_safe_name([f"plate{plate.id}", plate.name]))
                # Plot colony growth curves, ID map and time of appearance for each plate
                plots.plot_growth_curve([plate], save_path_plate)
                plots.plot_appearance_frequency([plate], save_path_plate, timestamps = image_files.timestamps_elapsed)
                plots.plot_appearance_frequency([plate], save_path_plate, timestamps = image_files.timestamps_elapsed, bar = True)

        if ANIMATION:
            # Plot individual plate images as an animation
            if not SILENT:
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
        if not SILENT:
            print("Unable to generate plots or animations from cached data. Run analysis on original images to generate plot images")

    if not SILENT:
        print(f"ColonyScanalyser analysis completed for: {BASE_PATH}")

    sys.exit()


if __name__ == "__main__":

    main()