def mm_to_pixels(millimeters, dots_per_inch = 300, pixels_per_mm = None):
    """
    Convert a measurement in millimetres to image pixels

    :param millimeters: the measurement to convert
    :param dots_per_inch: the conversion factor
    :param pixels_per_mm: optional conversion factor, instead of DPI
    """
    if millimeters <= 0 or dots_per_inch <= 0 or (pixels_per_mm is not None and pixels_per_mm <= 0):
        raise ValueError("All supplied arguments must be positive values")

    factor = dots_per_inch / 254

    if pixels_per_mm is not None:
        factor = pixels_per_mm

    return int(millimeters * factor)


def crop_image(image, crop_shape, center = None):
    """
    Get a subsection of an image

    Optionally specify a center point to crop around

    :param image: an image as a numpy array
    :param crop_shape: a row, column tuple array size to crop
    :param center: a row, column tuple co-ordinate point
    :returns: an image as a numpy array
    """
    import operator

    img = image.copy()

    if any(x < 0 for x in crop_shape) or len(image.shape) < len(crop_shape):
        raise ValueError(f"The crop shape ({crop_shape}) must be positive integers and the same dimensions as the image to crop")
    if crop_shape > img.shape:
        raise ValueError(f"The crop shape ({crop_shape}) cannot be larger than the image ({image.shape}) to crop")

    if center is None:
        # Use the center of the image
        start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, crop_shape))
    else:
        # Use a custom center point
        start = tuple(map(lambda a, da: a - da // 2, center, crop_shape))

    end = tuple(map(operator.add, start, crop_shape))
    
    if any(x < 0 for x in start) or end > img.shape:
        raise ValueError("The crop area cannot be outside the original image")

    slices = tuple(map(slice, start, end))

    return img[slices]

def cut_image_circle(image, center = None, radius = None, inverse = False):
    """
    Get the circular area of an image

    A center and radius can be specified to return a smaller part of the image

    :param image: an image as a numpy array
    :param center: a row, column tuple co-ordinate point
    :param radius: an integer length
    :param inverse: specify whether to return the inside or the outside area of the circle
    :returns: an image as a numpy array
    """
    import numpy as np

    img = image.copy()

    # Either use the entire image or crop to a radius
    if radius is None:
        radius = image.shape[0] // 2
    else:
        if any(radius * 2 > x for x in img.shape):
            raise ValueError("The circle radius cannot be larger than the supplied image")
        # Crop the image around the center point (if provided)
        crop_area = (radius * 2) + 1
        img = crop_image(img, (crop_area, crop_area), center)
        
    (cy, cx) = map(lambda x: x // 2, img.shape)
        
    # Calculate distances from center
    dist_x = np.vstack([(np.arange(img.shape[1]) - cx)] * (img.shape[0]))
    dist_y = np.vstack([(np.arange(img.shape[0]) - cy)] * (img.shape[1])).T
    dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
    
    if inverse:
        # Remove image information in area inside boundary
        img[dist <= radius] = 0
    else:
        # Remove image information in area outside boundary
        img[dist > radius] = 0

    return img


def get_image_circles(image, circle_radius, circle_count = -1, search_radius = 0):
    """
    Get circular parts of an image matching the size criteria

    :param image: a greyscale image as a numpy array
    :param circle_radius: a circle radius to search for, in pixels
    :param circle_count: the number of expected circles in the image
    :param search_radius: an additional area around the expected circle size to search
    :returns: a list of center co-ordinate tuples and radii
    """
    from skimage.transform import hough_circle, hough_circle_peaks
    from skimage.feature import canny
    from skimage.filters import threshold_otsu
    
    if image.size == 0:
        raise ValueError("The supplied image cannot be empty")
    img = image.copy()

    # Find edges in the image
    threshold = threshold_otsu(img)
    edges = canny(img < threshold, sigma = 3)

    # Check 10 pixels around the target radius, in steps of 5
    # Ignore search_area until hough_circle_peaks respects min_xdistance and min_ydistance
    # See: https://github.com/Erik-White/ColonyScanalyser/issues/10
    radii = range(circle_radius - 10, circle_radius + 10, 10)
    hough_circles = hough_circle(edges, radii)
    
    # Find the most significant circles
    _, cx, cy, radii = hough_circle_peaks(
        hough_circles,
        radii,
        min_xdistance = circle_radius,
        min_ydistance = circle_radius,
        total_num_peaks = circle_count
        )
        
    # Group and order coordinates in rows from top left
    coordinates = sorted(zip(cy, cx), key = lambda k: (round(k[0]/100), 0, k[1]))
    radii = [max(radii, key = radii.tolist().count) for x in radii]

    return [*zip(coordinates, radii)]


def remove_background_mask(image, mask, smoothing = 0.5, **filter_args):
    """
    Process an image by removing a background mask
    """
    from skimage.filters import gaussian
    
    if image.size == 0 or mask.size == 0:
        raise ValueError("The supplied image or mask cannot be empty")
    if image.shape != mask.shape:
        raise ValueError(f"The supplied image ({image.shape}) and mask ({mask.shape}) must be the same shape")
    image = image.copy()
    mask = mask.copy()
    
    # Do not process the image if it is empty
    if not image.any():
        return image

    # Get background mask intensity
    background = image[mask & (image > 0.05)].mean()
    
    # Determine image foreground
    ind = gaussian(image, smoothing, preserve_range = True, **filter_args) > background + 0.03
    
    # Subtract the mask, returning only the foreground
    return mask & ind


def watershed_separation(image, smoothing = 0.5):
    """
    Returns a labelled image where merged objects are separated
    """
    import numpy as np
    from scipy import ndimage
    from skimage.filters import gaussian
    from skimage.measure import label
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max

    if image.size == 0:
        return image
    img = image.copy()

    # Estimate smoothed distance from peaks to background
    distance = ndimage.distance_transform_edt(img)
    distance = gaussian(distance, smoothing)
    
    # Find image peaks, returned as a boolean array (indices = False)
    local_maxi = peak_local_max(distance, indices = False, footprint = np.ones((3, 3)), labels = img)

    # Find the borders around the peaks
    img = watershed(-distance, label(local_maxi), mask = img)

    return img