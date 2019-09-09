# Modules
import os
import sys
import glob
import argparse
import numpy as np
from operator import attrgetter
from itertools import izip
from skimage.io import imread, imsave
from skimage.color import rgb2grey
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import cm
import matplotlib.pyplot as plt


# Globals
lattice = (3, 2)


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
#Return only the filename by default
def separate_file_path(filepath, return_folderpath = False):
    (folderpath, filename) = os.path.split(os.path.abspath(filepath))
    if return_folderpath:
        return folderpath
    else:
        return filename


def get_subfoldername(data_folder, row, col):
    return data_folder+'_'.join(['row', str(row+1), 'col', str(col+1)])+'/'


def mkdir_plates(data_folder):
    '''Make subfolders for single plates'''
    for row in xrange(lattice[0]):
        for col in xrange(lattice[1]):
            subfn = get_subfoldername(data_folder, row, col)
            if not os.path.isdir(subfn):
                os.mkdir(subfn)


def find_plate_center_rough(im, lattice):
    '''Find the rough middle of the plate'''
    centers_y = np.linspace(0, 1, 2 * lattice[0] + 1)[1::2] * im.shape[0]
    centers_x = np.linspace(0, 1, 2 * lattice[1] + 1)[1::2] * im.shape[1]
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

#Finds all colonies on a plate and returns an array of co-ordinates(?)
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


# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyze the ScanLag data.')
    parser.add_argument('--verbose', type=int, default=0,
                       help='Output information level (0 to 4')
    parser.add_argument('--pos', type=int, nargs=2, default=None,
                        metavar=('ROW', 'COL'),
                        help='The row and column co-ordinates of the plate to study (default: all)')

    args = parser.parse_args()
    VERBOSE = args.verbose
    if args.pos != None:
        pos = map(lambda x: x - 1, args.pos)
    else:
        pos = None

    # Data locations
    data_folder = '~/Downloads/Scanlag/3112_1/'
    data_folder = os.path.expanduser(data_folder)
    fns = glob.glob(data_folder+'source_images/img*.tif')
    fns.sort()

    #Add a check and warning here if no images are loaded

    # Get info
    dates = [int(fn[:-8].split('_')[4]) for fn in fns]
    times = [int(fn[:-4].split('_')[-1]) for fn in fns]

    '''
    print 'times (minutes?):'
    print times[-1: -1 - len(areas): -1]
    sys.exit()
    '''

    # Divide plates and copy to separate files
    if VERBOSE >= 1:
        print 'Load plates'
    mkdir_plates(data_folder)
    plates_list = []
    for ifn, fn in enumerate(fns):
        d = dates[ifn]
        t = times[ifn]

        if VERBOSE >= 1:
            print 'Imaging date-time:', dates[ifn], '-', t
            print 'Check for split file(s)'
        has_all_split = True
        if pos is None:
            for row in xrange(lattice[0]):
                for col in xrange(lattice[1]):
                    if VERBOSE >= 2:
                        print 'row:', row, 'col:', col
                    fn_split = get_subfoldername(data_folder, row, col)+str(d)+'_'+str(t)+'.npy'
                    if not os.path.isfile(fn_split):
                        has_all_split = False
                        break
                if not has_all_split:
                    break
        else:
            (row, col) = pos
            fn_split = get_subfoldername(data_folder, row, col)+str(d)+'_'+str(t)+'.npy'
            if not os.path.isfile(fn_split):
                has_all_split = False

        if not has_all_split:
            if VERBOSE >= 1:
                print 'Split file not found, split now'

            if VERBOSE >= 1:
                print fn

            if VERBOSE >= 1:
                print 'Read image'
            im = imread(fn)

            if VERBOSE >= 1:
                print 'Convert to grey'
            img = rgb2grey(im)

            if VERBOSE >= 1:
                print 'Find plate center rough'
            centers = find_plate_center_rough(im, lattice)

            if VERBOSE >= 1:
                print 'Find plate borders'
            borders = find_plate_borders(img, centers)

            if VERBOSE >= 1:
                print 'Split image into plates'
            plates = split_image_into_plates(img, borders)

            if VERBOSE >= 1:
                print 'Save plates into separate images'
            for row in xrange(lattice[0]):
                for col in xrange(lattice[1]):
                    if VERBOSE >= 2:
                        print 'row:', row, 'col:', col
                    fn_split = get_subfoldername(data_folder, row, col)+str(d)+'_'+str(t)+'.npy'
                    roi = plates[row * lattice[1] + col]
                    roi.dump(fn_split)

            if pos is not None:
                (row, col) = pos
                plates = [plates[row * lattice[1] + col]]

        else:
            if VERBOSE >= 1:
                print 'Split file(s) found'

            plates = []
            if pos is None:
                for row in xrange(lattice[0]):
                    for col in xrange(lattice[1]):
                        if VERBOSE >= 2:
                            print 'row:', row, 'col:', col
                        fn_split = get_subfoldername(data_folder, row, col)+str(d)+'_'+str(t)+'.npy'
                        plates.append(np.load(fn_split, allow_pickle=True))
            else:
                (row, col) = pos
                fn_split = get_subfoldername(data_folder, row, col)+str(d)+'_'+str(t)+'.npy'
                plates.append(np.load(fn_split, allow_pickle=True))

        plates_list.append(plates)

    # Transpose the list of lists
    n_plates = len(plates_list[0])# The total number of plates
    plates_list = [[pl[n_plate] for pl in plates_list] for n_plate in xrange(n_plates)]
 
    # Segment
    if VERBOSE >= 1:
        print 'Segment'
    colonies_list = []

    for n_plate in xrange(n_plates):
        if VERBOSE >= 1:
            print 'Plate #'+str(n_plate + 1)

        plates = plates_list[n_plate]
        colonies_plates = []
        for it, plate in enumerate(plates):
            d = dates[it]
            t = times[it]
            if VERBOSE >= 2:
                print 'Date-time point', it+1, 'of', len(plates)

            plate_mask = plate > 0
            #Build a list of all colonies on the plate
            colonies = segment_image(plate, plate_mask)

            #Save list of colonies to disk
            filepath = "colony_data/time_point_"+str(d)+'_'+str(t)+".txt"
            create_folderpath(separate_file_path(filepath, True))
            with open(filepath, 'w') as outfile:
                for colony in colonies:
                    outfile.write(str(colony)+'\n')

            colonies_plates.append(colonies)

            if VERBOSE >= 2:
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                axs[0].imshow(plate)
                axs[1].imshow(colonies)
                fig.suptitle(str(dates[it])+'-'+str(t))
                imgpath = "plate_images/time_point_"+str(d)+'_'+str(t)+".jpg"
                create_folderpath(separate_file_path(imgpath, True))
                with open(imgpath, 'w') as outfile:
                    plt.savefig(outfile)

        colonies_list.append(colonies_plates)

    # Track
    # Start from last time point and proceed backwards by overlap (colonies do not move)
    if VERBOSE >= 1:
        print 'Tracking new colonies'
    
    #Loop through all the plates
    for index, colonies_plates in enumerate(colonies_list, start=1):
        if VERBOSE >= 1:
            print 'Plate', index, 'of', len(colonies_list)

        lineages = []

        #Loop through all colonies
        for i in xrange(1, colonies_plates[-1].max() + 1):
            #What is happening here?? Why is there a limit of 20 and FIXME?
            #The limit could be because this loop is too slow
            #Or does a higher number of colonies cause problems with excessive memory usage?
            '''
            # FIXME
            if i > 20:
                break
            '''
            #Limit to first n colonies for testing purposes
            if i > 1:
                break

            if VERBOSE >= 2:
                print 'Colony', i, 'of', colonies_plates[-1].max() + 1

            colony_final = colonies_plates[-1] == i
            area = colony_final.sum()
            print 'area=', area
            lineage = [colony_final]

            #Loop through all time points
            if VERBOSE >= 1:
                print 'Looping through', len(times), 'time points'

            for it in xrange(len(times) - 1):

                if VERBOSE >= 2:
                    print 'Time point', it, 'of', len(times)

                colonies = colonies_plates[-2 - it]
                ind = None
                overlap = 0
                
                if VERBOSE >= 3:
                    print 'Checking', len(colonies), 'co-ordinate rows(? maybe columns) at this time point for new appearances'

                for j in xrange(1, colonies.max() + 1):

                    if VERBOSE >= 4:
                        print 'Checking if colony', j, 'of', len(colonies), 'is new'

                    #Copy colony j in colonies collection
                    colony = colonies == j

                    #Intersect the two colony sets
                    #Then sum results to an integer
                    overlap_new = sum(np.in1d(colony, colony_final))
                    #Bitwise comparison might be faster
                    #But currently results in errors
                    #overlap_new = sum(colony&colony_final)
                    #overlap_new = sum(overlap_new)

                    #If they interset (overlap) then new colonies have appeared
                    if overlap_new > overlap:
                        ind = j
                        if overlap_new >= 0.5 * area:
                            break

                if ind is not None:
                    lineage.append(colonies == ind)
                else:
                    break

            lineages.append(lineage)

            #Dump lineages to check data
            for lineag in lineages:
                np.savetxt("lineages.txt", lineag, fmt='%s')

            # Plot colonies appearance??? against time point
            #This is probably colony max size at the end of the run!!
            if VERBOSE >= 3:
                ll = len(lineage)
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
                        axs[iax].imshow(lineage[0 -iax])
                        axs[iax].set_title(times[iax])
                    fig.suptitle('Colony '+str(i))
                    fig.show()
                    fig.savefig('testplot3.jpg')

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
            fig, ax = plt.subplots()
            for ia, areas in enumerate(areas_plates):
                # Reverse the order of the list
                ds = dates[-1: -1 - len(areas): -1]
                ts = times[-1: -1 - len(areas): -1]
                # Convert integer dates and times to Python datetimes
                from datetime import datetime
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