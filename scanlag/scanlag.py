# System modules
import sys
import glob
import math
import statistics
import argparse
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from operator import attrgetter
from distutils.util import strtobool

# Third party modules
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

# Local modules
import utilities
import file_access
import imaging
import plotting
import plots
import colony


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


def find_plate_center_rough(image, lattice):
    """
    Find the rough middle of a plate

    :param image: a black and white image as a numpy array
    :param lattice: a row, column tuple of the plate lattice size
    :returns: a list x, y tuples marking the plates centres
    """
    centers_y = np.linspace(0, 1, 2 * lattice[0] + 1)[1::2] * image.shape[0]
    centers_x = np.linspace(0, 1, 2 * lattice[1] + 1)[1::2] * image.shape[1]
    centers = [(int(cx), int(cy)) for cy in centers_y for cx in centers_x]

    return centers


def find_plate_borders(img, centers, refine = True):
    """
    Find the plate limits on the x and y axes

    :param img: a black and white image as a numpy array
    :param centers: a list x, y tuples marking the plates centres
    :param refine: repeat the process again for greater accuracy
    :returns: a list of min and max x/y values marking the plate edges
    """
    # Return a numpy array
    img = img.T

    borders = []
    for c in centers:
        # Slice the image around the center
        roi_c = img[c[0] - 50: c[0] + 50,
                        c[1] - 50: c[1] + 50]
                        
        iavg_c = np.mean(roi_c)
        istd_c = np.std(roi_c)
        dyn_r = img.max() - img.min()

        roi_side = 100
        th = 0.2 * dyn_r

        # Find x borders
        max_x = c[0]
        roi = img[max_x - roi_side // 2: max_x + roi_side // 2, c[1]]
        iavg = roi.mean()
        
        while np.abs(iavg - iavg_c) < th:
            max_x += 10
            roi = img[max_x - roi_side // 2: max_x + roi_side // 2, c[1]]
            iavg = roi.mean()

        min_x = c[0]
        roi = img[min_x - roi_side // 2: min_x + roi_side // 2, c[1]]
        iavg = roi.mean()
        while np.abs(iavg - iavg_c) < th:
            min_x -= 10
            roi = img[min_x - roi_side // 2: min_x + roi_side // 2, c[1]]
            iavg = roi.mean()

        # Find y borders
        max_y = c[1]
        roi = img[c[0], max_y - roi_side // 2: max_y + roi_side // 2]
        iavg = roi.mean()
        while np.abs(iavg - iavg_c) < th:
            max_y += 10
            roi = img[c[0], max_y - roi_side // 2: max_y + roi_side // 2]
            iavg = roi.mean()

        min_y = c[1]
        roi = img[c[0], min_y - roi_side // 2: min_y + roi_side // 2]
        iavg = roi.mean()
        while np.abs(iavg - iavg_c) < th:
            min_y -= 10
            roi = img[c[0], min_y - roi_side // 2: min_y + roi_side // 2]
            iavg = roi.mean()

        #Each plate has two points on each axis to mark the borders
        #The points are measured from the image edges
        borders.append([[min_x, max_x],
                        [min_y, max_y]])

    #if refine:
        #centers = list([map(lambda x: int(np.mean(x)), b) for b in borders])
        #borders = find_plate_borders(img.T, centers, refine=False)

    return borders


def split_image_into_plates(img, borders, edge_cut = 100):
    """
    Split image into lattice subimages and delete background
    
    :param img: a black and white image as a numpy array
    :param borders: a list of min and max x/y values marking the plate edges
    :param edge_cut: a radius, in pixels, to remove from the outer edge of the plate
    :returns: a list of plate images
    """
    plates = []

    for border in borders:
        #Find x and y centers, half way between the min/max values
        cx, cy = map(lambda x: int(np.mean(x)), border)
        radius = int(0.25 * (border[0][1] - border[0][0] + border[1][1] - border[1][0]) - edge_cut)

        # Copy a plate bounding box from the image
        plate_area = img[cy - radius: cy + radius + 1,
                  cx - radius: cx + radius + 1].copy()

        # Get a circular image
        plate_area = imaging.cut_image_circle(plate_area)          
        
        plates.append(plate_area)
    
    return plates


def segment_image(plate, plate_mask, plate_noise_mask, area_min = 5):
    """
    Finds all colonies on a plate and returns an array of co-ordinates

    If a co-ordinate is occupied by a colony, it contains that colonies labelled number

    :param plate: a black and white image as a numpy array
    :param mask: a black and white image as a numpy array
    :param plate_noise_mask: a black and white image as a numpy array
    :returns: a segmented and labelled image as a numpy array
    """
    from scipy import ndimage
    from skimage.morphology import remove_small_objects
    from skimage.measure import regionprops, label

    plate = imaging.remove_background_mask(plate, plate_mask)
    plate_noise_mask = imaging.remove_background_mask(plate_noise_mask, plate_mask)

    # Subtract an image of the first (i.e. empty) plate to remove static noise
    plate[plate_noise_mask] = 0

    # Fill any small gaps
    plate = ndimage.morphology.binary_fill_holes(plate)

    # Remove background noise
    plate = remove_small_objects(plate, min_size = area_min)

    #versions <0.16 do not allow for a mask
    #colonies = clear_border(pl_th, buffer_size = 1, mask = pl_th)

    colonies = label(plate)

    # Exclude objects that are too eccentric
    rps = regionprops(colonies, coordinates = "rc")
    for rp in rps:
        # Eccentricity of zero is a perfect circle
        # Circularity of 1 is a perfect circle
        circularity = (4 * math.pi * rp.area) / (rp.perimeter * rp.perimeter)

        if rp.eccentricity > 0.6 or circularity < 0.80:
            colonies[colonies == rp.label] = 0

    # Result is a 2D co-ordinate array
    # Each co-ordinate contains either zero or a unique colony number
    # The colonies are numbered from one to the total number of colonies on the plate
    return colonies


def segment_plate_timepoints(plate_images_list, date_times):
    """
    Build an array of segmented image data for all available time points

    Takes list of pre-processed plate images of size (total timepoints)

    :param plate_images_list: a list of black and white images as numpy arrays
    :param date_times: an ordered list of datetime objects
    :returns: a segmented and labelled list of images as numpy arrays
    :raises ValueError: if the size of plate_images_list and date_times do not match
    """
    # Check that the number of datetimes corresponds with the number of image timepoints
    if len(date_times) != len(plate_images_list):
        raise ValueError("Unable to process image timepoints. The supplied list of dates/times does not match the number of image timepoints")

    segmented_images = []
    plate_noise_mask = []
    # Loop through time points for the plate
    for i, plate_image in enumerate(plate_images_list, start=1):
        plate_mask = plate_image > 0
        # Create a noise mask from the first plate
        if i == 1:
            plate_noise_mask = plate_image
        # Build a 2D array of colony co-ordinate data for the plate image
        segmented_image = segment_image(plate_image, plate_mask, plate_noise_mask, area_min = 8)
        # segmented_images is an array of size (total plates)*(total timepoints)
        # Each time point element of the array contains a co-ordinate array of size (total image columns)*(total image rows)
        segmented_images.append(segmented_image)

    return segmented_images


def load_plate_timeline(load_filename, plate_lat, plate_pos = None):
    """
    Check if split image data is already stored and can be loaded
    """
    plate_list = dict()

    # Only load data for a single plate if it is specified
    if plate_pos is not None:
        row, column = plate_pos
        load_filepath = get_plate_directory(BASE_PATH, row, column).joinpath(load_filename)
        temp_data = file_access.load_file(load_filepath, file_access.CompressionMethod.LZMA, pickle = True)
        if temp_data is not None:
            plate_list[utilities.coordinate_to_index_number(plate_pos)] = temp_data

    # Otherwise, load data for all plates
    else:
        for row in range(1, plate_lat[0] + 1):
            for col in range(1, plate_lat[1] + 1):
                load_filepath = get_plate_directory(BASE_PATH, row, col).joinpath(load_filename)
                temp_data = file_access.load_file(load_filepath, file_access.CompressionMethod.LZMA, pickle = True)
                
                if temp_data is not None:
                    plate_list[(row - 1) * plate_lat[1] + col] = temp_data
                else:
                    # Do not return the list unless all elements were loaded sucessfully
                    plate_list = dict()
                    break
                
    return plate_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Analyze ScanLag images to track colonies and generate statistical data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("path", type = str,
                       help="Image files location", default = None)
    parser.add_argument("-v", "--verbose", type = int, default = 1,
                       help="Information output level")
    parser.add_argument("--plate_lattice", type = int, nargs = 2, default = (3, 2),
                        metavar = ("ROW", "COL"),
                        help="The row and column co-ordinate layout of plates. Example usage: --plate_lattice 3 3")
    parser.add_argument("--pos", "--plate_position", type = int, nargs = 2, default = argparse.SUPPRESS,
                        metavar = ("ROW", "COL"),
                        help = "The row and column co-ordinates of a single plate to study in the plate lattice. Example usage: --plate_position 2 1 (default: all)")
    parser.add_argument("--save_data", type = int, default = 1,
                        help = "Which compressed image data files to store on disk")
    parser.add_argument("--save_plots", type = int, default = 1,
                        help = "The detail level of plot images to store on disk")
    parser.add_argument("--use_saved", type = strtobool, default = True,
                        help = "Allow or prevent use of previously calculated data")

    args = parser.parse_args()
    BASE_PATH = args.path
    IMAGE_PATH = "source_images"
    VERBOSE = args.verbose
    PLATE_LATTICE = tuple(args.plate_lattice)
    if "plate_position" not in args:
        PLATE_POSITION = None
    else:
        PLATE_POSITION = args.plate_position
    SAVE_DATA = args.save_data
    SAVE_PLOTS = args.save_plots
    USE_SAVED = args.use_saved

    if PLATE_POSITION is not None:
        PLATE_POSITION = tuple(PLATE_POSITION)
        if utilities.coordinate_to_index_number(PLATE_POSITION) > utilities.coordinate_to_index_number(PLATE_LATTICE):
            raise ValueError("The supplied plate position coordinate is outside the plate grid")

    # Resolve working directory
    if BASE_PATH is None:
        raise ValueError("A path to a working directory must be supplied")
    else:
        BASE_PATH = Path(args.path).resolve()
    if not BASE_PATH.exists():
        raise ValueError("The supplied folder path could not be found:", BASE_PATH)
    if VERBOSE >= 1:
        print("Working directory:", BASE_PATH)

    # Find images in working directory
    image_formats = ["tif", "tiff", "png"]
    image_files = file_access.get_files_by_type(BASE_PATH, image_formats)
    # Try images directory if none found
    if not len(image_files) > 0:
        image_files = file_access.get_files_by_type(BASE_PATH.joinpath(IMAGE_PATH), image_formats)

    #Check if images have been loaded
    if not len(image_files) > 0:
        raise ValueError("No images could be found in the supplied folder path. Images are expected in these formats:", image_formats)

    # Move images to subdirectory if they are not already
    if IMAGE_PATH not in image_files[0].parts:
        image_files = file_access.move_to_subdirectory(image_files, IMAGE_PATH)
    if VERBOSE >= 1:
        print(len(image_files), "images found")
        
    # Get date and time information from filenames
    dates = [str(image.name[:-8].split("_")[-2]) for image in image_files]
    times = [str(image.name[:-4].split("_")[-1]) for image in image_files]
    
    # Convert string timestamps to Python datetime objects
    time_points = []
    time_points_elapsed = []
    for i, date in enumerate(dates):
        time_points.append(datetime.combine(datetime.strptime(dates[i], "%Y%m%d"),datetime.strptime(times[i], "%H%M").time()))
    # Also store time points as elapsed minutes since start
    for time_point in time_points:
        time_points_elapsed.append(int((time_point - time_points[0]).total_seconds() / 60))

    plates_list = dict()
    plates_list_segmented = dict()

    # Check if split and segmented image data is already stored and can be loaded
    segmented_image_data_filename = "split_image_data_segmented"
    if USE_SAVED:
        if VERBOSE >= 1:
            print("Attempting to load segmented processed image data for all plates")
        plates_list_segmented = load_plate_timeline(segmented_image_data_filename, PLATE_LATTICE, PLATE_POSITION)
    # Check that segmented image data has been loaded for all plates
    if len(plates_list_segmented) > 0:
        if len(plates_list_segmented.values()) == len(time_points):
            if VERBOSE >= 1:
                print("Successfully loaded segmented processed image data for all plates")
    else:
        if VERBOSE >= 1:
            print("Unable to load and uncompress segmented processed image data for all plates")

        # Check if split image data is already stored and can be loaded
        split_image_data_filename = "split_image_data"
        if USE_SAVED:
            if VERBOSE >= 1:
                print("Attempting to load and uncompress processed image data for all plates")
            plates_list_temp = load_plate_timeline(split_image_data_filename, PLATE_LATTICE, PLATE_POSITION)
            if plates_list_temp is not None:
                plates_list = plates_list_temp
        
        # Check that image data has been loaded for all plates
        if len(plates_list) > 0:
            if len(plates_list.values()) == len(time_points):
                if VERBOSE >= 1:
                    print("Successfully loaded processed image data for all plates")
        else:
            if VERBOSE >= 1:
                print("Unable to load processed image data for all plates")

            centers = None
            borders = None
            # Loop through and preprocess image files
            for ifn, image_file in enumerate(image_files):

                if VERBOSE >= 1:
                    print("Image number", ifn + 1, "of", len(image_files))

                if VERBOSE >= 2:
                    print("Imaging date-time:", time_points[ifn].strftime("%Y%m%d %H%M"))

                if VERBOSE >= 1:
                    print("Processing image:", image_file)
                img = imread(str(image_file), as_gray = True)
                
                if VERBOSE >= 2:
                    print("Find plate center rough")
                # Only find centers using first image. Assume plates do not move
                if centers is None:
                    centers = find_plate_center_rough(img, PLATE_LATTICE)

                if VERBOSE >= 2:
                    print("Find plate borders")
                # Only find borders using first image. Assume plates do not move
                if borders is None:
                    borders = find_plate_borders(img, centers)

                if VERBOSE >= 2:
                    print("Split image into plates")
                plates = split_image_into_plates(img, borders)

                if VERBOSE >= 2:
                    print("Store split plate image data for this time point")
                if PLATE_POSITION is not None:
                    # Save image for only a single plate
                    plate_index = utilities.coordinate_to_index_number(PLATE_POSITION)
                    if plate_index not in plates_list:
                        plates_list[plate_index] = list()
                    plates_list[plate_index].append(plates[plate_index - 1])
                else:
                    # Save images for all plates
                    for i, plate in enumerate(plates, start = 1):
                        # Store the image data from the current plate timepoint
                        if i not in plates_list:
                            plates_list[i] = list()
                        plates_list[i].append(plate)
                        
            if SAVE_DATA >= 2:
                # Save plates_list to disk for re-use if needed
                # Have to save each plate's data separately as it cannot be saved combined
                for i, plate in enumerate(plates_list.values(), start = 1):
                    if PLATE_POSITION is not None:
                        (row, col) = PLATE_POSITION
                    else:
                        (row, col) = utilities.index_number_to_coordinate(i, PLATE_LATTICE)
                    split_image_data_filepath = get_plate_directory(BASE_PATH, row, col, create_dir = True).joinpath(split_image_data_filename)
                    if VERBOSE >= 2:
                        print("Saving image data for plate #", i, "at position row", row, "column", col)
                    saved_status = file_access.save_file(split_image_data_filepath, plate, file_access.CompressionMethod.LZMA)
                    if VERBOSE >= 3:
                        if saved_status:
                            print("Saved processed image timeline data to:", split_image_data_filepath)
                        else:
                            print("An error occurred, unable to processed image timeline data to:", split_image_data_filepath)

        # Loop through plates and segment images at all timepoints
        for i, plate_timepoints in enumerate(plates_list.values(), start = 1):
            if PLATE_POSITION is not None:
                (row, col) = PLATE_POSITION
            else:
                (row, col) = utilities.index_number_to_coordinate(i, PLATE_LATTICE)
            if VERBOSE >= 2:
                print("Segmenting images from plate #", i, "at position row", row, "column", col)

            # plates_list is an array of size (total plates)*(total timepoints)
            # Each time point element of the array contains a co-ordinate array of size (total image columns)*(total image rows)
            segmented_plate_timepoints = segment_plate_timepoints(plate_timepoints, time_points)
            if segmented_plate_timepoints is None:
                print("Error: Unable to segment image data for plate")
                sys.exit()
            
            # Ensure labels remain constant
            segmented_plate_timepoints = imaging.standardise_labels_timeline(segmented_plate_timepoints)

            # Store the images for this plate
            plates_list_segmented[i] = segmented_plate_timepoints

            # Save segmented image plot for each timepoint
            if SAVE_PLOTS >= 2:
                for j, segmented_plate_timepoint in enumerate(segmented_plate_timepoints):
                    if VERBOSE >= 3:
                        print("Saving segmented image plot for time point ", j + 1, "of", len(segmented_plate_timepoints))
                    plots_path = file_access.create_subdirectory(BASE_PATH, "plots")
                    save_path = get_plate_directory(plots_path, row, col, create_dir = True)
                    save_path = file_access.create_subdirectory(save_path, "segmented_images")
                    image_path = plots.plot_plate_segmented(plate_timepoints[j], segmented_plate_timepoint, (row, col), time_points[j], save_path)
                    if image_path is not None:
                        if VERBOSE >= 3:
                            print("Saved segmented image plot to:", str(image_path))
                    else:
                        print("Error: Unable to save segmented image plot for plate at row", row, "column", col)

        # Save plates_list_segmented to disk for re-use if needed
        # Have to save data for each plate"s data separately
        # Can't pickle arrays with >3 dimensions
        if SAVE_DATA >= 1:
            for i, plate in enumerate(plates_list_segmented.values(), start = 1):
                if PLATE_POSITION is not None:
                    (row, col) = PLATE_POSITION
                else:
                    (row, col) = utilities.index_number_to_coordinate(i, PLATE_LATTICE)
                segmented_image_data_filepath = get_plate_directory(BASE_PATH, row, col, create_dir = True).joinpath(segmented_image_data_filename)
                if VERBOSE >= 2:
                    print("Saving segmented image data for plate #", i, "at position row", row, "column", col)
                saved_status = file_access.save_file(segmented_image_data_filepath, plate, file_access.CompressionMethod.LZMA)
                if VERBOSE >= 3:
                    if saved_status:
                        print("Saved processed and segmented image timeline data to:", segmented_image_data_filepath)
                    else:
                        print("An error occurred, unable to save processed and segmented image timeline data to:", segmented_image_data_filepath)
                        sys.exit()

    # Record individual colony information
    if VERBOSE >= 1:
        print("Tracking colonies")

    # Loop through plates and store data for each colony found
    from collections import defaultdict
    plate_colonies = defaultdict(dict)
    for i, plate_images in enumerate(plates_list_segmented.values(), start = 1):
        if VERBOSE >= 1:
            plate_number = i
            if PLATE_POSITION is not None:
                plate_number = utilities.coordinate_to_index_number(PLATE_POSITION)
            else:
                print("Tacking colonies on plate", plate_number, "of", len(plates_list_segmented))

        # Process image at each time point
        for j, plate_image in enumerate(plate_images):
            if VERBOSE >= 2:
                print("Tacking colonies at time point", j + 1, "of", len(plate_images))

            # Store data for each colony at every timepoint it is found
            plate_colonies[i] = colony.timepoints_from_image(plate_colonies[i], plate_image, time_points[j], time_points_elapsed[j])

        # Remove objects that do not have sufficient data points, usually just noise
        plate_colonies[i] = dict(filter(lambda elem: len(elem[1].timepoints) > len(time_points) * 0.2, plate_colonies[i].items()))
        # Remove object that do not show growth, these are not colonies
        plate_colonies[i] = dict(filter(lambda elem: elem[1].growth_rate > 1, plate_colonies[i].items()))

        if VERBOSE >= 1:
            print("Colony data stored for", len(plate_colonies[i].keys()), "colonies on plate", plate_number)

    # Store pickled data to allow quick re-use
    if SAVE_DATA >= 1:
        save_path = BASE_PATH.joinpath("processed_data")
        save_status = file_access.save_file(save_path, plate_colonies, file_access.CompressionMethod.LZMA)
        if VERBOSE >= 1:
            if save_status:
                print(f"Cached data saved to {save_path}")
            else:
                print(f"An error occurred and cached data could not be written to disk at {save_path}")

    # Plot colony growth curves and time of appearance for the plate
    if SAVE_PLOTS >= 2:
        for plate_id, plate in plate_colonies.items():
            if PLATE_POSITION is not None:
                (row, col) = PLATE_POSITION
            else:
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
        print("Scanlag analysis complete")

    sys.exit()