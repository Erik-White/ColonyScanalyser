# Modules
import os
import sys
import glob
import argparse
import numpy as np
from datetime import datetime
from operator import attrgetter
from itertools import izip
from skimage.io import imread, imsave
from skimage.color import rgb2grey
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import cm
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt

# Local modules
from sci_utilities import is_outlier




# Globals
# plate_lattice is the shape of the plate arrangement in rows*columns
plate_lattice = (3, 2)


# Functions
# Create any folder that is needed in a filepath
def create_folderpath(filepath):
    # Exception handling is required in Python2 to protect against race conditions
    try: 
        os.makedirs(filepath)
    except OSError:
        if not os.path.isdir(filepath):
            raise
    # In Python3 this can be reduced to
    # os.makedirs(path, exist_ok=True)


#Separates the folderpath and filename from a filepath
#Return the filename by default
def separate_filepath(filepath, return_folderpath = False):
    (folderpath, filename) = os.path.split(os.path.abspath(filepath))
    if return_folderpath:
        return folderpath
    else:
        return filename


def get_subfoldername(data_folder, row, col):
    return data_folder+'_'.join(['row', str(row), 'col', str(col)])+os.path.sep


# Checks whether a file exists and contains data
def file_exists(filepath):
    if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
        return True


# Return a row and column index for a plate
# Plate and lattice co-ordinate numbers are 1-based
def plate_number_to_coordinates(plate_number, lattice):
    row = ((plate_number - 1) // lattice[1]) + 1
    col = ((plate_number - 1) % lattice[1]) + 1
    return (row, col)


def mkdir_plates(data_folder):
    '''Make subfolders for single plates'''
    for row in xrange(plate_lattice[0]):
        for col in xrange(plate_lattice[1]):
            subfn = get_subfoldername(data_folder, row + 1, col + 1)
            if not os.path.isdir(subfn):
                os.mkdir(subfn)


# Creates an empty list of lists with the shape required for the number of plates
def init_plate_image_list(init_list, lattice):
    for i in xrange(1, np.prod(lattice)):
        init_list.append(list())
    return init_list

def find_plate_center_rough(image, lattice):
    '''Find the rough middle of the plate'''
    centers_y = np.linspace(0, 1, 2 * lattice[0] + 1)[1::2] * image.shape[0]
    centers_x = np.linspace(0, 1, 2 * lattice[1] + 1)[1::2] * image.shape[1]
    centers = [(int(cx), int(cy)) for cy in centers_y for cx in centers_x]
    return centers


def find_plate_borders(img, centers, refine=True):
    '''Find the plates'''
    # Images are sick objects, they have Y and then X!
    img = img.T

    borders = []
    for c in centers:

        roi_c = img[c[0] - 50: c[0] + 50,
                    c[1] - 50: c[1] + 50]

        iavg_c = roi_c.mean()
        istd_c = roi_c.std()
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

    if refine:
        centers = [map(lambda x: int(np.mean(x)), b) for b in borders]
        borders = find_plate_borders(img.T, centers, refine=False)

    return borders


def split_image_into_plates(img, borders, edge_cut=100):
    '''Split image into lattice subimages and delete background'''
    plates = []
    for border in borders:
        #Find x and y centers, measured from image edges
        (cx, cy) = map(lambda x: int(np.mean(x)), border)
        radius = int(0.25 * (border[0][1] - border[0][0] + border[1][1] - border[1][0]) - edge_cut)
        #Copy the image in a radius around the center point
        roi = img[cy - radius: cy + radius + 1,
                  cx - radius: cx + radius + 1].copy()

        (cy, cx) = map(lambda x: x / 2.0, roi.shape)
        dist_x = np.vstack([(np.arange(roi.shape[1]) - cx)] * (roi.shape[0]))
        dist_y = np.vstack([(np.arange(roi.shape[0]) - cy)] * (roi.shape[1])).T
        dist = np.sqrt(dist_x**2 + dist_y**2)
        roi[dist > radius] = 0
        
        plates.append(roi)
    
    return plates

# Finds all colonies on a plate and returns an array of co-ordinates
# If a co-ordinate is occupied by a colony, it contains that colonies unique ID number
def segment_image(plate, plate_mask, area_min=30, area_max=500):
    '''Segment the image based on simple thresholding'''
    from skimage.measure import regionprops, label
    from skimage.filters import gaussian

    bg = plate[plate_mask & (plate > 0.05)].mean()
    plate_gau = gaussian(plate, 0.5)
    ind = plate_gau > bg + 0.03
    pl_th = plate_mask & ind

    ## Naive segmentation
    #colonies = label(pl_th)

    # Watershed on smoothed distance from bg to segment merged colonies
    # (they are round)
    from scipy import ndimage
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max
    distance = ndimage.distance_transform_edt(pl_th)
    distance = gaussian(distance, 0.5)
    # Find image peaks and return a boolean array (indices-False)
    # Peaks are represented by True values
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=pl_th)
    markers = label(local_maxi)
    # Find the borders around the peaks
    colonies = watershed(-distance, markers, mask=pl_th)

    # Exclude objects that are too eccentric
    from skimage.measure import regionprops
    rps = regionprops(colonies)
    for i, rp in enumerate(rps, start=1):
        if rp.eccentricity > 0.6:
            colonies[colonies == i] = 0
    # Return an ordered array, relabelled sequentially
    from skimage.segmentation import relabel_sequential
    (colonies, fwdmap, revmap) = relabel_sequential(colonies)

    # Instead of this randomizing section, could maybe use lab2rgb to paint random colours over?
    #https://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.label2rgb

    
    # Randomize colors for clarity
    ind = np.arange(colonies.max()) + 1
    np.random.shuffle(ind)
    #Return an array of zeros with the same shape as the input array
    colonies_random = np.zeros_like(colonies)
    #Replace zero values with actual values
    for i, ii in enumerate(ind, start=1):
        colonies_random[colonies == i] = ii
        
    # Result is a 2D co-ordinate array
    # Each co-ordinate contains either zero or a unique colony number
    # The colonies are numbered from one to the total number of colonies on the plate
    return colonies_random

    
# Build an array of segmented image data for all available time points
# Takes list of pre-processed plate images of size (total timepoints)
# Also requires a list of Python datetimes corresponding to the image timepoints
def segment_plate_timepoints(plate_images_list, date_times):

    # Check that the number of datetimes corresponds with the number of image timepoints
    if len(date_times) != len(plate_images_list):
        print 'Unable to save segmented image plot. The supplied list of dates/times does not match the number of image timepoints'
        return None

    segmented_images = []
    # Loop through time points for the plate
    for i, plate_image in enumerate(plate_images_list, start=1):
        plate_mask = plate_image > 0
        # Build a 2D array of colony co-ordinate data for the plate image
        segmented_image = segment_image(plate_image, plate_mask)
        # segmented_images is an array of size (total plates)*(total timepoints)
        # Each time point element of the array contains a co-ordinate array of size (total image columns)*(total image rows)
        segmented_images.append(segmented_image)
    return segmented_images


# Save the processed plate images and corresponding segmented data plots
# Images are stored in the corresponding plate data folder i.e. /row_2_col_1/segmented_image_plots/
# A Python datetime is required to save the image with the correct filename
def save_plate_segmented_image(plate_image, segmented_image, plate_row, plate_column, date_time):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(plate_image)
    axs[1].imshow(segmented_image)
    fig_title = ' '.join(["Plate at row", str(plate_row), ": column", str(plate_column), "at time point", date_time.strftime("%Y/%m/%d %H:%M")])
    fig.suptitle(fig_title)
    folder_path = get_subfoldername(data_folder, plate_row, plate_column) + "segmented_image_plots" + os.path.sep
    image_path = ''.join([folder_path, "time_point_", str(date_time.strftime("%Y%m%d")), '_' + str(date_time.strftime("%H%M")), ".jpg"])
    create_folderpath(folder_path)
    with open(image_path, 'w') as outfile:
        plt.savefig(outfile)
    plt.close()

    # Return the path to the new image if it was saved successfully
    if os.path.isfile(image_path):
        return image_path
    else:
        return None


# Check if split image data is already stored and can be loaded
def load_plate_timeline(plate_list, load_filename, plate_lat, plate_pos = None):
    if plate_pos is not None:
        load_filepath = get_subfoldername(data_folder, plate_pos[0], plate_pos[1]) + os.path.sep + load_filename
        if file_exists(load_filepath):
            plate_list[0].append(np.load(load_filepath, allow_pickle=True))
    else:
        for row in xrange(plate_lattice[0]):
            for col in xrange(plate_lattice[1]):
                load_filepath = get_subfoldername(data_folder, row + 1, col + 1) + os.path.sep + load_filename
                if file_exists(load_filepath):
                    plate_list[row * plate_lattice[1] + col].append(np.load(load_filepath, allow_pickle=True))
                else:
                    print "Unable to load stored image data for plate on row", row + 1, "col", col + 1
                    plate_list = None
                    break
    # Do not return the list unless all elements were loaded sucessfully
    if plate_list is not None:
        return plate_list


# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyze the ScanLag data.')
    parser.add_argument('--verbose', type=int, default=0,
                       help='Output information level (0 to 4')
    parser.add_argument('--pos', type=int, nargs=2, default=None,
                        metavar=('ROW', 'COL'),
                        help='The row and column co-ordinates of the plate to study (default: all)')
    # Add a new argument to specify if stored data/images should be recalculated and overwritten

    args = parser.parse_args()
    VERBOSE = args.verbose
    if args.pos != None:
        PLATE_POSITION = args.pos
    else:
        PLATE_POSITION = None

    # Data locations
    data_folder = '~/Downloads/Scanlag/3112_1/'
    data_folder = os.path.expanduser(data_folder)
    image_filenames = glob.glob(data_folder+'source_images/img*.tif')
    image_filenames.sort()

    #Check if images have been loaded
    if len(image_filenames) is None:
        print "No images found in the specified filepath: "
        print data_folder
        sys.exit()
    elif VERBOSE >= 1:
        print len(image_filenames), "images found"

    # Get date and time information from filenames
    dates = [int(image_filename[:-8].split('_')[4]) for image_filename in image_filenames]
    times = [int(image_filename[:-4].split('_')[-1]) for image_filename in image_filenames]
    # Convert integer timestamps to Python datetime objects
    time_points = []
    for i, date in enumerate(dates):
        time_points.append(datetime.combine(datetime.strptime(str(date), '%Y%m%d'),datetime.strptime(str(times[i]), '%H%M').time()))

    # Initialise plates_list with empty lists to the total number of plates
    plates_list = list()
    segmented_images = list()
    if PLATE_POSITION is None:
        n_lattice = (1, 1)
    else:
        n_lattice = np.prod(plate_lattice)
        
    plates_list = init_plate_image_list(plates_list, n_lattice)
    segmented_images = init_plate_image_list(segmented_images, n_lattice)

    # Check if split and segmented image data is already stored and can be loaded
    segmented_images = load_plate_timeline(segmented_images, "split_image_data_segmented.npz", plate_lattice, PLATE_POSITION)

    # Check that segmented image data has been loaded for all plates
    if ((PLATE_POSITION is not None) and (len(segmented_images) != 1)) or (len(segmented_images) != np.prod(plate_lattice)):

        # Check if split image data is already stored and can be loaded
        plates_list = load_plate_timeline(plates_list, "split_image_data.npz", plate_lattice, PLATE_POSITION)

        # Check that image data has been loaded for all plates
        if ((PLATE_POSITION is not None) and (len(plates_list) != 1)) or len(plates_list) != np.prod(plate_lattice):

            # Divide plates and copy to separate files
            if VERBOSE >= 1:
                print 'Create plate subfolders'
            mkdir_plates(data_folder)

            # Loop through image files
            for ifn, image_filename in enumerate(image_filenames):
                d = dates[ifn]
                t = times[ifn]

                if VERBOSE >= 1:
                    print 'Imaging date-time:', dates[ifn], '-', t

                if VERBOSE >= 1:
                    print image_filename

                if VERBOSE >= 1:
                    print 'Read image'
                im = imread(image_filename)

                if VERBOSE >= 1:
                    print 'Convert to grey'
                img = rgb2grey(im)

                if VERBOSE >= 1:
                    print 'Find plate center rough'
                centers = find_plate_center_rough(im, plate_lattice)

                if VERBOSE >= 1:
                    print 'Find plate borders'
                borders = find_plate_borders(img, centers)

                if VERBOSE >= 1:
                    print 'Split image into plates'
                plates = split_image_into_plates(img, borders)

                if VERBOSE >= 1:
                    print 'Store split plate image data for this time point'
                if PLATE_POSITION is not None:
                    # Save image for only a single plate
                    plates_list[0].append(plates[np.prod(plate_lattice)-1])
                else:
                    # Save images for all plates
                    for i, plate in enumerate(plates):
                        # Store the image data from the current plate timepoint
                        plates_list[i].append(plate)

            # Save plates_list to disk for re-use if needed
            # Have to save each plate's data separately as it cannot be saved combined
            for i, plate in enumerate(plates_list):
                (row, col) = plate_number_to_coordinates(i + 1, plate_lattice)
                split_image_data_filepath = get_subfoldername(data_folder, row, col) + os.path.sep + split_image_data_filename
                if VERBOSE >= 2:
                    print "Saving image data for plate #", i + 1, "at position row", row, "column", col
                with open(split_image_data_filepath, 'w') as outfile:
                    np.savez(split_image_data_filepath, plates_list[i])

        '''
        # Transpose plates_list to an array of size (total plates)*(total timepoints)
        # Store the total number of plates as n_plates
        n_plates = len(plates_list[0])
        plate_images_list = [[pl[n_plate] for pl in plates_list] for n_plate in xrange(n_plates)]
        '''

        # Loop through plates and segment images at all timepoints
        for i, plate_timepoints in enumerate(plates_list):
            if VERBOSE >= 2:
                print 'Segmenting images from plate #', i
            
            # Loop through each timepoint
            for j, plate_timepoint in plate_timepoints:
                if VERBOSE >= 3:
                    print "Segmenting image data for time point ", j + 1, "of", len(plate_timepoint)

                # segmented_images is an array of size (total plates)*(total timepoints)
                # Each time point element of the array contains a co-ordinate array of size (total image columns)*(total image rows)
                segmented_plate = segment_plate_timepoints(plate_timepoint, time_points)
                segmented_images[i].append(segmented_plate)

                if VERBOSE >= 2:
                    (row, col) = plate_number_to_coordinates(i + 1, plate_lattice)
                    image_path = save_plate_segmented_image(plate_timepoint, segmented_plate, plate_row, plate_column, time_points[j])
                    if image_path is not None:
                        if VERBOSE >= 3:
                            print "Saved segmented image plot to:", image_path
                    else:
                        print "Error: Unable to save segmented image plot for plate at row", row, "column", col

        '''''
        Give segmented_images a name that better reflects that it is all plates and all images
        '''''
        print "segmented_images len=", len(segmented_images)


        # Save segmented_images to disk for re-use if needed
        # Have to save each plate's data separately as it cannot be saved combined
        for i, plate in enumerate(segmented_images):
            (row, col) = plate_number_to_coordinates(i + 1, plate_lattice)
            segmented_image_data_filepath = get_subfoldername(data_folder, row, col) + os.path.sep + split_image_data_filename
            if VERBOSE >= 2:
                print "Saving segmented image data for plate #", i + 1, "at position row", row, "column", col
            np.savez(segmented_image_data_filepath, segmented_images[i])

    # Track
    # Start from last time point and proceed backwards by overlap (colonies do not move)
    if VERBOSE >= 1:
        print 'Tracking new colonies'
    
    # Loop through all the plates
    for index, colonies_plates in enumerate(segmented_images, start=1):
        if VERBOSE >= 1:
            print 'Plate', index, 'of', len(segmented_images)

        lineages = []
        colonies_total = colonies_plates[-1].max() + 1

        # Loop to the maximum unique colony id number
        for i in xrange(1, colonies_total):
            #What is happening here?? Why is there a limit of 20 and FIXME?
            #The limit could be because this loop is too slow
            #Or does a higher number of colonies cause problems with excessive memory usage?
            '''
            # FIXME
            if i > 20:
                break
            '''
            #Limit to first n co-ordinate points for testing purposes
            if i > 1:
                break

            if VERBOSE >= 2:
                print 'Colony', i, 'of', colonies_total

            # Store the data for the final time point
            # Perform an elemntwise comparison with the current colony id number
            # Produces a boolean array
            colony_final = colonies_plates[-1] == i
            # Sum the number of True co-ordinate points to give an area approximation
            colony_area_final = colony_final.sum()
            print 'colony area final=', colony_area_final
            # Store the boolean co-ordinate array
            lineage = [colony_final]

            if VERBOSE >= 1:
                print 'Looping through', len(times), 'time points'

            # Loop through all time points
            for it in xrange(len(times) - 1):

                if VERBOSE >= 2:
                    print 'Time point', it, 'of', len(times)

                print "colonies_plates first dim (timepoints)=", len(colonies_plates)
                print "colonies_plates second dim (y? axis)=", len(colonies_plates[-1][-1])
                print "colonies_plates third dim (x? axis)=", len(colonies_plates[-1][-1][-1])
                print colonies_plates[-1]
                print colonies_plates[-1][-1]
                print colonies_plates[-1][-1][-1]
                # Store the co-ordinate data at the preceeding time point
                #colonies = colonies_plates[-2 - it]
                colonies = list(reversed(colonies_plates[it]))
                print "colonies first dim=", len(colonies)
                print "colonies second dim=", len(colonies[-1])
                print colonies[-1]
                print colonies[-1][-1]
                sys.exit()
                ind = None
                overlap = 0
                
                '''
                if VERBOSE >= 3:
                    print 'Checking', len(colonies), 'co-ordinate rows(? maybe columns) at this time point for new appearances'

                # Loop through image co-ordinate data
                for j in xrange(1, colonies.max() + 1):

                    if VERBOSE >= 4:
                        print 'Checking co-ordinate point', j, 'of', len(colonies)

                    # Perform an elementwise comparision with j
                    colony = colonies == j

                    # Intersect the two colony sets, then sum results to an integer
                    # This compares if a colony at the current time point is present at the final time point
                    overlap_new = sum(np.in1d(colony, colony_final))
                    print 'overlap_new=', overlap_new

                    #Bitwise comparison might be faster
                    #But currently results in errors
                    #overlap_new = sum(colony&colony_final)
                    #overlap_new = sum(overlap_new)

                    #If they interset (overlap) then new colonies have appeared
                    if overlap_new > overlap:
                        ind = j
                        if overlap_new >= 0.5 * area:
                            break


            lineages.append(lineage)
                '''
                # Perform an elemntwise comparison with the current colony id number
                colony = colonies == i
                print 'colony == i ', colony
                # Intersect the two colony sets, then sum results to an integer
                # This compares if a colony at the current time point is present at the final time point
                overlap_new = sum(np.in1d(colony, colony_final))
                print 'overlap_new=', overlap_new

                #If they interset (overlap) then new colonies have appeared
                if overlap_new > overlap:
                    ind = True
                    if overlap_new >= 0.5 * colony_area_final:
                        break

                if ind is not None:
                    lineages.append(colony)
                else:
                    break
            print lineages

            # Plot colonies appearance??? against time point
            #This is probably colony max size at the end of the run!!
            if VERBOSE >= 3:
                ll = len(lineage)
                print 'lineage length=', len(lineage)
                # Check that some lineages have been recorded
                if ll > 0:
                    fig, axs = plt.subplots(1, ll, figsize=(2 + 4 * ll, 6))
                    for iax in xrange(len(axs)):
                        axs[iax].imshow(lineage[-1 -iax])#-1 =last element of the array
                        axs[iax].set_title(times[iax])
                    fig.suptitle('Colony '+str(i))
                    fig.show()
                    fig.savefig('testplot2.jpg')

                    # This should be colony first appearance
                    fig, axs = plt.subplots(1, ll, figsize=(2 + 4 * ll, 6))
                    for iax in xrange(len(axs)):
                        for enumer in lineage:
                            axs[iax].imshow(enumer)#0 = first element of the array
                        axs[iax].set_title(times[iax])
                    fig.suptitle('Colony '+str(i))
                    fig.show()
                    fig.savefig('testplot3.jpg')


        #Dump lineages to check data
        for lint, linea in enumerate(lineages):
            filepath = "lineages/lineage_"+str(lint)+".txt"
            create_folderpath(separate_filepath(filepath, True))
            with open(filepath, 'w') as outfile:
                outfile.write(str(linea)+'\n')

        #numpy.savetxt("data", numpy.array([x, y]).T, header="x y")
        print 'lineages length', len(lineages)
        #np.savetxt("lineages.txt", lineages, header="x y")

        '''
        # Write the array to disk
        with open('lineages.txt', 'w') as outfile:
            # I'm writing a header here just for the sake of readability
            # Any line starting with "#" will be ignored by numpy.loadtxt
            outfile.write('# Array shape: {0}\n')

            # Iterating through a ndimensional array produces slices along
            # the last axis. This is equivalent to data[i,:,:] in this case
            for data_slice in lineages:

                # Writing out a break to indicate different slices...
                outfile.write('# New slice\n')

                for data_slice2 in data_slice:
                    # The formatting string indicates that I'm writing out
                    # the values in left-justified columns 7 characters in width
                    # with 2 decimal places.  
                    #np.savetxt(outfile, data_slice2, fmt='%-7.2f')
                    np.savetxt(outfile, data_slice2)

                    # Writing out a break to indicate different slices...
                    outfile.write('# New slice2\n')
        '''

        areas_plates = [map(np.sum, lineage) for lineage in lineages]
        if VERBOSE >= 1:
            # Filter outliers from area data
            # "~" operates as a logical not operator on boolean numpy arrays
            #areas_plates = areas_plates[~is_outlier(areas_plates)]
            fig, ax = plt.subplots()
            for ia, areas in enumerate(areas_plates):
                # Reverse the order of the list
                ds = dates[-1: -1 - len(areas): -1]
                ts = times[-1: -1 - len(areas): -1]
                # Convert integer dates and times to Python datetimes
                from matplotlib.dates import  DateFormatter
                timez = []
                for timerator, time in enumerate(ts, start=0):
                    timez.append(datetime.combine(datetime.strptime(str(ds[timerator]), "%Y%m%d") ,datetime.strptime(str(time),"%H%M").time()))
                #ax.plot(ts, areas, lw=2, c=cm.jet(1.0 * ia / len(areas_plates)))
                ax.xaxis.set_major_formatter(DateFormatter("%b-%d %H:%M"))
                ax.plot_date(timez, areas, lw=2, c=cm.jet(1.0 * ia / len(areas_plates)))
            
            ax.set_xlabel('Date-Time')
            ax.set_ylabel('Area [px^2]')
            ax.set_yscale('log')
            fig.suptitle("Colony #"+str(i)+"total area")#Need to find actual colony id number
            fig.autofmt_xdate()
            
    plt.savefig('testplot.jpg')

    plt.ion()
    plt.show()