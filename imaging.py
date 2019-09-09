# Return the circular area of an image
# A center and radius can be specified to return a smaller part of the image
def cut_image_circle(image, **kwargs):
    import numpy as np
    img = image.copy()
    center = kwargs.get('center', None)
    radius = kwargs.get('radius', None)
    inverse = kwargs.get('inverse', False)

    # Use the centre of the image bounding box if none specified
    if center is None:
        (cy, cx) = map(lambda x: x / 2.0, img.shape)
    else:
        (cy, cx) = center
    if radius is None:
        radius = len(image[0]) / 2

    dist_x = np.vstack([(np.arange(img.shape[1]) - cx)] * (img.shape[0]))
    dist_y = np.vstack([(np.arange(img.shape[0]) - cy)] * (img.shape[1])).T
    dist = np.sqrt(dist_x**2 + dist_y**2)

    if inverse:
        # Remove image information in area inside boundary
        img[dist < radius] = 0
    else:
        # Remove image information in area outside boundary
        img[dist > radius] = 0

    return img
    

# Checks a circular area around each labelled part of an image
# If other objects are found in the radius, remove the label
def clear_merged_labels(segmented_image, threshold_radius = 2):
    import numpy as np
    from skimage.measure import regionprops
    img = segmented_image.copy()

    for rp in regionprops(segmented_image):
        radius = (rp.equivalent_diameter / 2) + threshold_radius

        # Copy the image in a radius around the center point
        search_area = cut_image_circle(segmented_image, center = rp.centroid, radius = radius)
        # Remove labelled section if it is merged
        # Should only find 0 and one other label
        if len(np.unique(search_area)) > 2:
            img[img == rp.label] = 0

    return img


# Process an image by removing a background mask
def remove_background_mask(image, mask):
    from skimage import exposure
    from skimage.filters import gaussian
    
    background = image[mask & (image > 0.05)].mean()
    ind = gaussian(image, 0.5) > background + 0.03

    return mask & ind


# Returns a labelled image where merged objects are separated
def watershed_separation(image, smoothing = 0.5):
    img = image.copy()
    # Estimate smoothed distance from peaks to background
    distance = ndimage.distance_transform_edt(img)
    distance = gaussian(distance, smoothing)
    
    # Find image peaks, returned as a boolean array (indices = False)
    local_maxi = peak_local_max(distance, indices = False, footprint = np.ones((3, 3)), labels = img)

    # Find the borders around the peaks
    img = watershed(-distance, label(local_maxi), mask = img)

    return img



# Replace labels on similar images to allow tracking over time
# Labels should supplied as a 2D list of label, co-ordinate pairs
def standardise_labels_timeline(images_list, label_prepend = 99, start_at_end = True):
    images = list(images_list)
    if start_at_end:
        images.reverse()
        
    for i, image in enumerate(images):
        if i > 0:
            label_prepend = None

        # Store the labels for the current image
        labels = get_labelled_centers(image, label_prepend = label_prepend)
        
        # Apply labels to all subsequent images
        for j in xrange(i, len(images)):
            images[j] = replace_image_point_labels(images[j], labels)

    if start_at_end:
        images.reverse()

    return images


# Build a list of labels and their centers
def get_labelled_centers(image, label_prepend = None):
    from skimage.measure import regionprops

    rps = regionprops(image)
    labels = []

    # Loop through labelled regions and store information
    for rp in rps:
        label = str(rp.label)
        if label_prepend is not None:
            label  = str(label_prepend) + label 
        labels.append((int(label), rp.centroid))
    
    return labels


# Replace the labelled at a list of points with new labels
# Labels should supplied as a 2D list of label, co-ordinate pairs
def replace_image_point_labels(image, labels):
    img = image.copy()
    for label, point in labels:
        # Ensure point values are integers
        point = tuple(int(x) for x in point)
        # Find the existing label at the point
        index = img[point]
        # Replace the existing label with new, excluding background
        if index > 0:
            img[img == index] = label

    return img