# vim: fdm=indent
'''
author:     Fabio Zanini
date:       03/08/14
content:    Test for scanner.
'''
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

        print c[0]
        print type(c[0])
        print c[1]
        print type(c[1])
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
        (cx, cy) = map(lambda x: int(np.mean(x)), border)
        radius = int(0.25 * (border[0][1] - border[0][0] + border[1][1] - border[1][0]) - edge_cut)
        print radius
        roi = img[cy - radius: cy + radius + 1,
                  cx - radius: cx + radius + 1].copy()
        (cy, cx) = map(lambda x: x / 2.0, roi.shape)
        dist_x = np.vstack([(np.arange(roi.shape[1]) - cx)] * (roi.shape[0]))
        dist_y = np.vstack([(np.arange(roi.shape[0]) - cy)] * (roi.shape[1])).T
        dist = np.sqrt(dist_x**2 + dist_y**2)
        roi[dist > radius] = 0
        
        plates.append(roi)
    
    return plates


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
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                labels=pl_th)
    markers = label(local_maxi)
    colonies = watershed(-distance, markers, mask=pl_th)

    # Exclude objects that are too eccentric
    from skimage.measure import regionprops
    rps = regionprops(colonies)
    for i, rp in enumerate(rps, 1):
        if rp.eccentricity > 0.6:
            colonies[colonies == i] = 0
    from skimage.segmentation import relabel_sequential
    (colonies, fwdmap, revmap) = relabel_sequential(colonies)

    # Randomize colors for clarity
    ind = np.arange(colonies.max()) + 1
    np.random.shuffle(ind)
    colonies_random = np.zeros_like(colonies)
    for i, ii in enumerate(ind, 1):
        colonies_random[colonies == i] = ii

    return colonies_random



# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyze the ScanLag data.')
    parser.add_argument('--verbose', type=int, default=0,
                       help='Verbosity level')
    parser.add_argument('--pos', type=int, nargs=2, default=None,
                        metavar=('ROW', 'COL'),
                        help='ROW COL of plate to study (default: all)')

    args = parser.parse_args()
    VERBOSE = args.verbose
    if args.pos != None:
        pos = map(lambda x: x - 1, args.pos)
    else:
        pos = None

    # Data locations
    data_folder = '~/Downloads/Scanlag/3112_1/'
    data_folder = os.path.expanduser(data_folder)
    fns = glob.glob(data_folder+'img*.tif')
    fns.sort()

    # Get info
    times = [int(fn[:-4].split('_')[-1]) for fn in fns]

    # Divide plates and copy to separate files
    if VERBOSE >= 1:
        print 'Load plates'
    mkdir_plates(data_folder)
    plates_list = []
    for ifn, fn in enumerate(fns):
        t = times[ifn]

        if VERBOSE >= 1:
            print 'Time:', t
            print 'Check for split file(s)'
        has_all_split = True
        if pos is None:
            for row in xrange(lattice[0]):
                for col in xrange(lattice[1]):
                    if VERBOSE >= 2:
                        print 'row:', row, 'col:', col
                    fn_split = get_subfoldername(data_folder, row, col)+str(t)+'.npy'
                    if not os.path.isfile(fn_split):
                        has_all_split = False
                        break
                if not has_all_split:
                    break
        else:
            (row, col) = pos
            fn_split = get_subfoldername(data_folder, row, col)+str(t)+'.npy'
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
                    fn_split = get_subfoldername(data_folder, row, col)+str(t)+'.npy'
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
                        fn_split = get_subfoldername(data_folder, row, col)+str(t)+'.npy'
                        plates.append(np.load(fn_split, allow_pickle=True))
            else:
                (row, col) = pos
                fn_split = get_subfoldername(data_folder, row, col)+str(t)+'.npy'
                plates.append(np.load(fn_split, allow_pickle=True))

        plates_list.append(plates)

    # Transpose the list of lists
    n_plates = len(plates_list[0])
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
            t = times[it]
            if VERBOSE >= 2:
                print 'Time point #'+str(it + 1)

            plate_mask = plate > 0
            colonies = segment_image(plate, plate_mask)

            colonies_plates.append(colonies)

            if VERBOSE >= 2:
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                axs[0].imshow(plate)
                axs[1].imshow(colonies)
                fig.suptitle(str(t))

        colonies_list.append(colonies_plates)

    # Track
    # Start from last time point and proceed backwards by overlap (colonies do not move)
    if VERBOSE >= 1:
        print 'Track'
    
    for colonies_plates in colonies_list:
        lineages = []
        for i in xrange(1, colonies_plates[-1].max() + 1):
            # FIXME
            if i > 20:
                break

            if VERBOSE >= 2:
                print 'Colony', i

            colony_final = colonies_plates[-1] == i
            area = colony_final.sum()
            lineage = [colony_final]
            for it in xrange(len(times) - 1):
                colonies = colonies_plates[-2 - it]
                ind = None
                overlap = 0
                for j in xrange(1, colonies.max() + 1):
                    colony = colonies == j
                    #Intersect the two colony sets
                    #Then sum results to an integer
                    overlap_new = sum(np.in1d(colony, colony_final))
                    if overlap_new > overlap:
                        ind = j
                        if overlap_new >= 0.5 * area:
                            break

                if ind is not None:
                    lineage.append(colonies == ind)
                else:
                    break

            lineages.append(lineage)

            if VERBOSE >= 3:
                ll = len(lineage)
                fig, axs = plt.subplots(1, ll, figsize=(2 + 4 * ll, 6))
                #Getting error: axs has no length
                for iax in xrange(len(axs)):
                    axs[iax].imshow(lineage[-1 -iax])
                    axs[iax].set_title(times[iax])
                fig.suptitle('Colony '+str(i))

        areas_plates = [map(np.sum, lineage) for lineage in lineages]
        if VERBOSE >= 1:
            fig, ax = plt.subplots()
            for ia, areas in enumerate(areas_plates):
                ts = times[-1: -1 - len(areas): -1]
                ax.plot(ts, areas, lw=2,
                        c=cm.jet(1.0 * ia / len(areas_plates)))
            ax.set_xlabel('Time [min]')
            ax.set_ylabel('Area [px^2]')
            ax.set_yscale('log')

    plt.ion()
    plt.show()