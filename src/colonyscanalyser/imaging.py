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


def rgb_to_name(color_rgb, color_spec = "css3"):
    """
    Convert an RGB tuple to the closest named web colour

    For a full list of colours, see: https://www.w3.org/TR/css-color-3/

    :param color_rgb: a red, green, blue colour value tuple
    :param color_spec: a color spec from the webcolors module
    :returns: a named colour string
    """
    from webcolors import hex_to_rgb

    # Default to CSS3 color spec if none specified
    color_spec = str.lower(color_spec)
    if color_spec == "html4":
        from webcolors import html4_hex_to_names
        color_dict = html4_hex_to_names
    elif color_spec == "css2":
        from webcolors import css2_hex_to_names
        color_dict = css2_hex_to_names
    elif color_spec == "css21":
        from webcolors import css21_hex_to_names
        color_dict = css21_hex_to_names
    else:
        from webcolors import css3_hex_to_names
        color_dict = css3_hex_to_names

    min_colours = {}
    for key, name in color_dict.items():
        r_c, g_c, b_c = hex_to_rgb(key)
        rd = (r_c - color_rgb[0]) ** 2
        gd = (g_c - color_rgb[1]) ** 2
        bd = (b_c - color_rgb[2]) ** 2
        min_colours[(rd + gd + bd)] = name

    return min_colours[min(min_colours.keys())]


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

    if any(x < 0 for x in crop_shape) or any(not isinstance(x, int) for x in crop_shape) or len(image.shape) < len(crop_shape):
        raise ValueError(
            f"The crop shape ({crop_shape}) must be positive integers and the same dimensions as the image to crop"
            )
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
        if any(radius * 2 > x for x in img.shape[:2]):
            raise ValueError("The circle radius cannot be larger than the supplied image")
        # Crop the image around the center point
        crop_area = (int(radius) * 2) + 1
        img = crop_image(img, (crop_area, crop_area), center)

    (cy, cx) = map(lambda x: x // 2, img.shape[:2])

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
    from numpy import inf
    from skimage.transform import hough_circle, hough_circle_peaks
    from skimage.feature import canny
    from skimage.filters import threshold_otsu

    if circle_count == -1:
        circle_count = inf

    if image.size == 0:
        raise ValueError("The supplied image cannot be empty")
    img = image.copy()

    # Find edges in the image
    threshold = threshold_otsu(img)
    edges = canny(img < threshold, sigma = 3)

    # Check 2 * search_radius pixels around the target radius, in steps of 10
    radii = range(circle_radius - search_radius, circle_radius + search_radius, 10)
    hough_circles = hough_circle(edges, radii)

    # Find the most significant circles
    _, cx, cy, radii = hough_circle_peaks(
        hough_circles,
        radii,
        min_xdistance = circle_radius,
        min_ydistance = circle_radius
        # total_num_peaks = circle_count
        )

    # Temporary helper function until hough_circle_peaks respects min distances
    cx, cy, radii = circles_radius_median(cx, cy, radii, circle_count)

    # Create a Dict with y values as keys and row numbers as values
    # Allows a quick lookup of the row values for sorting
    row_groups = dict()
    row_count = 0
    for i, x in enumerate(sorted(cy)):
        if i == 0 or abs((sorted(cy)[i - 1]) - x) > circle_radius:
            row_count += 1
        row_groups[x] = row_count

    # Group and order coordinates in rows from top left
    coordinates = sorted(zip(cy, cx), key = lambda k: (row_groups[k[0]], 0, k[1]))
    radii = [max(radii, key = radii.tolist().count) for x in radii]

    return [*zip(coordinates, radii)]


def circles_radius_median(cx, cy, radii, circle_count):
    """
    Temp fix to exclude overlapping circles
    Assumes circle_count circles can be found at radius median
    hough_circle_peaks currently ignores min_xdistance and min_ydistance
    See: https://github.com/Erik-White/ColonyScanalyser/issues/10
    """
    from numpy import inf

    if circle_count == inf:
        circle_count = len(radii) + 1

    # hough_circle_peaks are returned sorted by peak intensity
    # Find the most common radius from the 'best' circles
    radius_median = max(radii[slice(circle_count)], key = radii.tolist().count)

    # Keep only the circles with the median radius
    cx = cx[radii == radius_median]
    cy = cy[radii == radius_median]
    radii = radii[radii == radius_median]

    return (cx, cy, radii)


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

    # Find image peaks, returned as a boolean array
    local_maxi = peak_local_max(distance, indices = False, footprint = np.ones((3, 3)), labels = img)

    # Find the borders around the peaks
    img = watershed(-distance, label(local_maxi), mask = img)

    return img