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
import matplotlib
matplotlib.use("TkAgg") # Required for OSX
from matplotlib import cm
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt

# Local modules
import utilities
import file_access
import imaging
import plotting
import colony


def get_plate_directory(parent_path, row, col, create_dir = False):
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
    """Find the rough middle of the plate"""
    centers_y = np.linspace(0, 1, 2 * lattice[0] + 1)[1::2] * image.shape[0]
    centers_x = np.linspace(0, 1, 2 * lattice[1] + 1)[1::2] * image.shape[1]
    centers = [(int(cx), int(cy)) for cy in centers_y for cx in centers_x]

    return centers


def find_plate_borders(img, centers, refine=True):
    """Find the plates"""
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


def split_image_into_plates(img, borders, edge_cut=100):
    """Split image into lattice subimages and delete background"""
    plates = []

    for border in borders:
        #Find x and y centers, measured from image edges
        (cx, cy) = map(lambda x: int(np.mean(x)), border)
        radius = int(0.25 * (border[0][1] - border[0][0] + border[1][1] - border[1][0]) - edge_cut)

        # Copy a plate bounding box from the image
        plate_area = img[cy - radius: cy + radius + 1,
                  cx - radius: cx + radius + 1].copy()

        # Get a circular image
        plate_area = imaging.cut_image_circle(plate_area)          
        
        plates.append(plate_area)
    
    return plates


def segment_image(plate, plate_mask, plate_noise_mask, area_min=30, area_max=500):
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
    plate = remove_small_objects(plate, min_size = 10)

    # Return an ordered array, relabelled sequentially
    #(colonies, fwdmap, revmap) = relabel_sequential(colonies)

    #versions <0.16 do not allow for a mask
    #colonies = clear_border(pl_th, buffer_size = 1, mask = pl_th)

    colonies = label(plate)

    # Exclude objects that are too eccentric
    rps = regionprops(colonies)
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
        segmented_image = segment_image(plate_image, plate_mask, plate_noise_mask)
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
            plate_list[0] = temp_data

    # Otherwise, load data for all plates
    else:
        for row in range(PLATE_LATTICE[0]):
            for col in range(PLATE_LATTICE[1]):
                load_filepath = get_plate_directory(BASE_PATH, row + 1, col + 1).joinpath(load_filename)
                temp_data = file_access.load_file(load_filepath, file_access.CompressionMethod.LZMA, pickle = True)
                if temp_data is not None:
                    plate_list[row * PLATE_LATTICE[1] + col] = temp_data
                else:
                    # Do not return the list unless all elements were loaded sucessfully
                    plate_list = dict()
                    break
                
    return plate_list


# Save the processed plate images and corresponding segmented data plots
# Images are stored in the corresponding plate data folder i.e. /row_2_col_1/segmented_image_plots/
# A Python datetime is required to save the image with the correct filename
def save_plate_segmented_image(plate_image, segmented_image, plate_coordinate, date_time):
    """
    Saves processed plate images and corresponding segmented data plots

    :param plate_image: a black and white image as a numpy array
    :param segmented_image: a segmented and labelled image as a numpy array
    :param plate_coordinate: a row, column tuple representing the plate position
    :param date_time: a datetime object
    :returns: the filepath string if the plot is sucessfully saved
    """
    from skimage.measure import regionprops
    (plate_row, plate_column) = plate_coordinate

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(plate_image)
    # Set colour range so all colonies are clearly visible and the same colour
    ax[1].imshow(segmented_image, vmax = 1)
            
    # Place maker labels on colonies
    for rp in regionprops(segmented_image):
        ax[1].annotate("+",
            plotting.rc_to_xy(rp.centroid),
            xycoords = "data",
            color = "red",
            horizontalalignment = "center",
            verticalalignment = "center"
            )
        ax[1].annotate(str(rp.label),
            plotting.rc_to_xy(rp.centroid),
            xytext = (5, -16),
            xycoords = "data",
            textcoords = "offset pixels",
            color = "white",
            alpha = 0.8,
            )

    fig_title = " ".join(["Plate at row", str(plate_row), ": column", str(plate_column), "at time point", date_time.strftime("%Y/%m/%d %H:%M")])
    fig.suptitle(fig_title)

    folder_path = get_plate_directory(BASE_PATH, plate_row, plate_column).joinpath("segmented_image_plots")
    image_path = "".join(["time_point_", str(date_time.strftime("%Y%m%d")), "_" + str(date_time.strftime("%H%M")), ".jpg"])
    folder_path.mkdir(parents = True, exist_ok = True)

    with open(folder_path.joinpath(image_path), "wb") as outfile:
        plt.savefig(outfile)
    plt.close()

    # Return the path to the new image if it was saved successfully
    if Path.is_file(folder_path.joinpath(image_path)):
        return image_path
    else:
        return None


# Script
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze ScanLag images to track colonies and generate statistical data.")
    parser.add_argument("--path", type=str, default=None,
                       help="Image files location")
    parser.add_argument("--verbose", type=int, default=0,
                       help="Output information level")
    parser.add_argument("--plate_lattice", type=int, nargs=2, default=(3, 2),
                        metavar=("ROW", "COL"),
                        help="The row and column co-ordinate layout of plates")
    parser.add_argument("--plate_position", type=int, nargs=2, default=None,
                        metavar=("ROW", "COL"),
                        help="The row and column co-ordinates of the plate to study (default: all)")
    parser.add_argument("--save_data", type=int, default=1,
                        help="Which data files to store on disk")
    parser.add_argument("--save_plots", type=int, default=1,
                        help="The level of plot images to store on disk")
    parser.add_argument("--use_saved", type=strtobool, default=True,
                        help="Allow or prevent use of previously calculated data")

    args = parser.parse_args()
    BASE_PATH = args.path
    IMAGE_PATH = "source_images"
    VERBOSE = args.verbose
    PLATE_LATTICE = args.plate_lattice
    PLATE_POSITION = args.plate_position
    SAVE_DATA = args.save_data
    SAVE_PLOTS = args.save_plots
    USE_SAVED = args.use_saved

    if PLATE_POSITION is not None:
        PLATE_POSITION = tuple(PLATE_POSITION)

    # Resolve working directory
    if BASE_PATH is None:
        # Default to user home directory if none supplied
        BASE_PATH = Path.home()
    else:
        BASE_PATH = Path(args.path).resolve()
    if not BASE_PATH.exists():
        raise ValueError("The supplied folder path could not be found:", BASE_PATH)
    if VERBOSE >= 1:
        print("Working directory:", BASE_PATH)

    # Find images in working directory
    image_files = file_access.get_files_by_type(BASE_PATH, "tif, png")
    # Try images directory if none found
    if not len(image_files) > 0:
        image_files = file_access.get_files_by_type(BASE_PATH.joinpath(IMAGE_PATH), "tif, png")

    #Check if images have been loaded
    if not len(image_files) > 0:
        raise ValueError("No images could be found in the supplied folder path. Image are expected in .tif or .png format")

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
    if USE_SAVED:
        if VERBOSE >= 1:
            print("Attempting to load segmented processed image data for all plates")
        segmented_image_data_filename = "split_image_data_segmented1"
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
        if USE_SAVED:
            if VERBOSE >= 1:
                print("Attempting to load and uncompress processed image data for all plates")
            split_image_data_filename = "split_image_data"
            plates_list_temp = load_plate_timeline(split_image_data_filename, PLATE_LATTICE, PLATE_POSITION)
            if plates_list_temp is not None:
                plates_list = plates_list_temp
        
        # Check that image data has been loaded for all plates
        if len(plates_list) > 0:
            if len(plates_list[0]) == len(time_points):
                if VERBOSE >= 1:
                    print("Successfully loaded processed image data for all plates")
        else:
            if VERBOSE >= 1:
                print("Unable to load processed image data for all plates")
                
            # Divide plates and copy to separate files
            if VERBOSE >= 1:
                print("Create plate subfolders")

            centers = None
            borders = None
            # Loop through image files
            for ifn, image_file in enumerate(image_files):

                if VERBOSE >= 1:
                    print("Image number", ifn + 1, "of", len(image_files))

                if VERBOSE >= 2:
                    print("Imaging date-time:", time_points[ifn].strftime("%Y%m%d %H%M"))

                if VERBOSE >= 1:
                    print("Read image:", image_file)
                img = imread(str(image_file), as_gray = True)
                
                if VERBOSE >= 1:
                    print("Find plate center rough")
                # Only find centers using first image. Assume plates do not move
                if centers is None:
                    centers = find_plate_center_rough(img, PLATE_LATTICE)

                if VERBOSE >= 1:
                    print("Find plate borders")
                # Only find borders using first image. Assume plates do not move
                if borders is None:
                    borders = find_plate_borders(img, centers)

                if VERBOSE >= 1:
                    print("Split image into plates")
                plates = split_image_into_plates(img, borders)

                if VERBOSE >= 1:
                    print("Store split plate image data for this time point")
                if PLATE_POSITION is not None:
                    # Save image for only a single plate
                    if 0 not in plates_list:
                        plates_list[0] = list()
                    plates_list[0].append(plates[np.prod(PLATE_POSITION)-1])
                else:
                    # Save images for all plates
                    for i, plate in enumerate(plates):
                        # Store the image data from the current plate timepoint
                        if i not in plates_list:
                            plates_list[i] = list()
                        plates_list[i].append(plate)
                        
            if SAVE_DATA >= 2:
                # Save plates_list to disk for re-use if needed
                # Have to save each plate's data separately as it cannot be saved combined
                for i, plate in enumerate(plates_list.values()):
                    (row, col) = utilities.index_number_to_coordinate(i + 1, PLATE_LATTICE)
                    split_image_data_filepath = get_plate_directory(BASE_PATH, row, col, create_dir = True).joinpath(split_image_data_filename)
                    if VERBOSE >= 2:
                        print("Saving image data for plate #", i + 1, "at position row", row, "column", col)
                    saved_status = file_access.save_file(split_image_data_filepath, plate, file_access.CompressionMethod.LZMA)
                    if VERBOSE >= 3:
                        if saved_status:
                            print("Saved processed image timeline data to:", split_image_data_filepath)
                        else:
                            print("An error occurred, unable to processed image timeline data to:", split_image_data_filepath)

        # Loop through plates and segment images at all timepoints
        for i, plate_timepoints in enumerate(plates_list.values()):
            (row, col) = utilities.index_number_to_coordinate(i + 1, PLATE_LATTICE)
            if VERBOSE >= 2:
                print("Segmenting images from plate #", i + 1, "at position row", row, "column", col)

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
                    image_path = save_plate_segmented_image(plate_timepoints[j], segmented_plate_timepoint, (row, col), time_points[j])
                    if image_path is not None:
                        if VERBOSE >= 3:
                            print("Saved segmented image plot to:", image_path)
                    else:
                        print("Error: Unable to save segmented image plot for plate at row", row, "column", col)

        # Save plates_list_segmented to disk for re-use if needed
        # Have to save data for each plate"s data separately
        # Can't pickle arrays with >3 dimensions
        if SAVE_DATA >= 1:
            for i, plate in enumerate(plates_list_segmented.values()):
                (row, col) = utilities.index_number_to_coordinate(i + 1, PLATE_LATTICE)
                segmented_image_data_filepath = get_plate_directory(BASE_PATH, row, col).joinpath(segmented_image_data_filename)
                if VERBOSE >= 2:
                    print("Saving segmented image data for plate #", i + 1, "at position row", row, "column", col)
                saved_status = file_access.save_file(segmented_image_data_filepath, plate, file_access.CompressionMethod.LZMA)
                if VERBOSE >= 3:
                    if saved_status:
                        print("Saved processed and segmented image timeline data to:", segmented_image_data_filepath)
                    else:
                        print("An error occurred, unable to save processed and segmented image timeline data to:", segmented_image_data_filepath)

    # Record individual colony information
    if VERBOSE >= 1:
        print("Tracking colonies")

    # Loop through plates
    from collections import defaultdict
    plate_colonies = defaultdict(dict)
    for i, plate_images in enumerate(plates_list_segmented.values()):
        if VERBOSE >= 1:
            print("Tacking colonies on plate", i + 1, "of", len(plates_list_segmented))

        # Process image at each time point
        for j, plate_image in enumerate(plate_images):
            if VERBOSE >= 2:
                print("Tacking colonies at time point", j + 1, "of", len(plate_images))

            plate_colonies[i] = colony.timepoints_from_image(plate_colonies[i], plate_image, time_points[j])

        print("number of plates = ", len(plate_colonies))
        print("number of colonies for plate 1", len(plate_colonies[0].keys()))
        print("number of timepoints for plate 1, colony 50", len(plate_colonies[0][50].timepoints))

    sys.exit()



    # Track
    # Start from last time point and proceed backwards by overlap (colonies do not move)
    if VERBOSE >= 1:
        print("Tracking colonies")

    # Loop through plates
    plate_colony_lineages = []
    from collections import defaultdict
    plate_colony_areas = defaultdict(list)
    for i, plate_images in enumerate(plates_list_segmented):
        if VERBOSE >= 1:
            print("Tacking colonies on plate", i + 1, "of", len(plates_list_segmented))

        # Loop backwards through plate timepoints
        segmented_image_final = plate_images[-1]
        for j, segmented_image in enumerate(reversed(plate_images)):
            if VERBOSE >= 2:
                print("Tacking colonies at time point", j + 1, "of", len(plate_images))
                
            overlap = segmented_image&segmented_image_final

            (uniques, counts) = np.unique(overlap, return_counts = True)
            # Store area values using colony ids as dictionary keys
            # Do not include "colony 0", ie areas with no colonies
            for k, colony_id in enumerate(uniques):
                # Elimate noise
                if colony_id > 0 and counts[k] > 10:
                    plate_colony_areas[colony_id].append((list(reversed(time_points_elapsed))[j], counts[k]))

        # Time of appearance
        time_of_appearance = dict()
        for colony_id, values in plate_colony_areas.items():
            timepoints, areas = zip(*values)
            time_of_appearance[colony_id] = min(timepoints)

        # Number of colonies appearing at each time point
        if SAVE_PLOTS >= 1:
            # Initialize a dictionary that contains all time points
            # Bar plot fails if values aren"t initialised to zero and left as None
            #time_points_dict = dict.fromkeys(time_points_elapsed, 0)
            time_points_dict = dict()

            fig, ax = plt.subplots()
            #fig, ax = plt.subplots(figsize=(10,8))
            colors = iter(cm.rainbow(np.linspace(0, 1, len(time_of_appearance))))

            # Plot areas for each colony
            for colony_id, timepoint in time_of_appearance.items():
                # Map areas to a full dictionary of timepoints
                if timepoint not in time_points_dict:
                    time_points_dict[timepoint] = 0
                time_points_dict[timepoint] += 1
            """   
            for key, value in time_points_dict.items():
                if value <= 2 or key > 1600:
                    time_points_dict.pop(key)
            """

            # Use zip to return a sorted list of tuples (key, value) from the dictionary
            bars = ax.bar(*zip(*sorted(time_points_dict.items())),
                width = 10
                #color = next(colors),
                #marker = "o",
                #label = str(colony_id)
                )
            for key, value in time_points_dict.items():
                ax.text(x = key, y = value + 0.2, s = str(key // 60) + " hours", color="blue")
            #ax.set_xlim([min(time_points_dict.keys()), max(time_points_dict.keys())])
            ax.set_xlabel("Elapsed time (minutes)")
            ax.set_ylabel("Number of colonies")
            fig.suptitle("Colony appearances over time")
            fig.show()
            fig.savefig("plots/testplot_appearance_frequency.jpg")

        # Plot change in colony area for this plate
        if SAVE_PLOTS >= 3:
            # Initialize a dictionary that contains all time points
            time_points_dict = dict.fromkeys(time_points_elapsed)

            fig, ax = plt.subplots()
            #fig, ax = plt.subplots(figsize=(10,8))
            colors = iter(cm.rainbow(np.linspace(0, 1, len(time_points_dict))))

            # Plot areas for each colony
            for colony_id in plate_colony_areas.keys():
                # Map areas to a full dictionary of timepoints
                for (timepoint, area) in plate_colony_areas[colony_id]:
                    time_points_dict[timepoint] = area
                    
                # Use zip to return a sorted list of tuples (key, value) from the dictionary
                ax.scatter(*zip(*sorted(time_points_dict.items())),
                    #color = next(colors),
                    marker = "o",
                    label = str(colony_id)
                    )
            ax.set_yscale("log")
            ax.set_xlabel("Elapsed time (minutes)")
            ax.set_ylabel("Area [px^2]")
            fig.suptitle("Colony areas over time")
            fig.show()
            fig.savefig("plots/testplot_areas.jpg")

    sys.exit()