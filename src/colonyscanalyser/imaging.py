from typing import Optional, Union, Tuple, List
from numpy import ndarray
from skimage.transform._geometric import GeometricTransform, SimilarityTransform


def mm_to_pixels(millimeters: float, dots_per_inch: float = 300, pixels_per_mm: Optional[float] = None) -> float:
    """
    Convert a measurement in millimetres to image pixels

    :param millimeters: the measurement to convert
    :param dots_per_inch: the conversion factor
    :param pixels_per_mm: optional conversion factor, instead of DPI
    :returns: a value in pixels
    """
    if millimeters <= 0 or dots_per_inch <= 0 or (pixels_per_mm is not None and pixels_per_mm <= 0):
        raise ValueError("All supplied arguments must be positive values")

    factor = dots_per_inch / 25.4

    if pixels_per_mm is not None:
        factor = pixels_per_mm

    return int(millimeters * factor)


def rgb_to_name(color_rgb: Union[Tuple[int, int, int], Tuple[float, float, float]], color_spec: str = "css3") -> str:
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
        from webcolors import HTML4_HEX_TO_NAMES
        color_dict = HTML4_HEX_TO_NAMES
    elif color_spec == "css2":
        from webcolors import CSS2_HEX_TO_NAMES
        color_dict = CSS2_HEX_TO_NAMES
    elif color_spec == "css21":
        from webcolors import CSS21_HEX_TO_NAMES
        color_dict = CSS21_HEX_TO_NAMES
    else:
        from webcolors import CSS3_HEX_TO_NAMES
        color_dict = CSS3_HEX_TO_NAMES

    min_colours = {}
    for key, name in color_dict.items():
        r_c, g_c, b_c = hex_to_rgb(key)
        rd = (r_c - color_rgb[0]) ** 2
        gd = (g_c - color_rgb[1]) ** 2
        bd = (b_c - color_rgb[2]) ** 2
        min_colours[(rd + gd + bd)] = name

    return min_colours[min(min_colours.keys())]


def crop_image(
    image: ndarray,
    crop_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    center: Optional[Tuple[int, int]] = None
) -> ndarray:
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


def cut_image_circle(
    image: ndarray,
    center: Optional[Tuple[float, float]] = None,
    radius: Optional[float] = None,
    inverse: bool = False,
    background_color = 0
) -> ndarray:
    """
    Get the circular area of an image

    A center and radius can be specified to return a smaller part of the image

    :param image: an image as a numpy array
    :param center: a row, column tuple co-ordinate point
    :param radius: an integer length
    :param inverse: if True, returns the image area outside the circle
    :param background_color: the color to replace the empty parts of the image
    :returns: an image as a numpy array
    """
    from numpy import sqrt, arange, vstack

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

    cy, cx = map(lambda x: x // 2, img.shape[:2])

    # Calculate distances from center
    dist_x = vstack([(arange(img.shape[1]) - cx)] * (img.shape[0]))
    dist_y = vstack([(arange(img.shape[0]) - cy)] * (img.shape[1])).T
    dist = sqrt(dist_x ** 2 + dist_y ** 2)

    if inverse:
        # Remove image information in area inside boundary
        img[dist <= radius] = background_color
    else:
        # Remove image information in area outside boundary
        img[dist > radius] = background_color

    return img


def get_image_circles(
    image: ndarray,
    circle_radius: float,
    circle_count: int = -1,
    search_radius: float = 0
) -> List[Tuple[Tuple[float, float], float]]:
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
        min_ydistance = circle_radius,
        total_num_peaks = circle_count
    )

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


def image_as_rgb(image: ndarray) -> ndarray:
    """
    Convert an image in any colour mode to RGB

    :param image: an image as a numpy array
    :returns: the image with RGB colour
    """
    from skimage.color import rgba2rgb, gray2rgb

    # If the image has no colour channels, it must be greyscale
    if len(image.shape) < 3 or image.shape[-1] == 1:
        return gray2rgb(image)

    # Remove alpha channel if present
    if image.shape[-1] == 4:
        image = rgba2rgb(image)

    return image


def remove_background_mask(image: ndarray, smoothing: float = 1, sigmoid_cutoff: float = 0.4, **filter_args) -> ndarray:
    """
    Separate the image foreground from the background

    Returns a boolean mask of the image foreground

    :param image: an image as a numpy array
    :param smoothing: a sigma value for the gaussian filter
    :param sigmoid_cutoff: cutoff for the sigmoid exposure function
    :param filter_args: arguments to pass to the gaussian filter
    :returns: a boolean image mask of the foreground
    """
    from skimage import img_as_bool
    from skimage.exposure import adjust_sigmoid
    from skimage.filters import gaussian, threshold_triangle

    if image.size == 0:
        raise ValueError("The supplied image cannot be empty")

    image = image.astype("float64", copy = True)

    # Do not process the image if it is empty
    if not image.any():
        return img_as_bool(image)

    # Apply smoothing to reduce noise
    image = gaussian(image, smoothing, **filter_args)

    # Heighten contrast
    image = adjust_sigmoid(image, cutoff = sigmoid_cutoff, gain = 10)

    # Find background threshold and return only foreground
    return image > threshold_triangle(image, nbins = 10)