def round_tuple_floats(tuple_item, precision = 2):
    if not isinstance(tuple_item, tuple):
        raise ValueError(f"The object must be of type 'tuple', not type '{type(tuple_item)}'")
    return tuple(map(lambda x: isinstance(x, float) and round(x, precision) or x, tuple_item))


def index_number_to_coordinate(index, lattice):
    """
    Calculate row and column numbers for an item index

    Lattice co-ordinate and item index numbers are 1-based

    :param index: item index integer
    :param lattice: row and column tuple boundaries
    :returns: row and column co-ordinate tuple
    :raises IndexError: if the returned index number would exceed the lattice size
    """
    (lattice_row, lattice_col) = lattice
    if index < 1 or lattice_row < 1 or lattice_col < 1:
        raise ValueError("All supplied parameters must be greater than zero")

    row = ((index - 1) // lattice[1]) + 1
    col = ((index - 1) % lattice[1]) + 1

    if row > lattice_row or col > lattice_col:
        raise IndexError("Index number is greater than the supplied lattice size")

    return (row, col)


def coordinate_to_index_number(coordinate):
    """
    Find a positional index for a coordinate

    Starting along rows and then down columns
    """
    import numpy as np

    return np.prod(coordinate)


def progress_bar(bar_progress, bar_length = 30, message = ""):
    """
    Output a simple progress bar to the console

    :param bar_progress: the overall progress as a percentage
    :param bar_length: the display length of the progress bar
    :param message: text to display next to the progress bar
    """
    from sys import stdout

    # Reset cursor to beginning of the line
    stdout.write('\r')
    
    # Write to the line
    stdout.write(f"[{'#' * int(bar_length * (bar_progress / 100)):{bar_length}s}] {int(bar_progress)}% {message}")
    
    # If the bar is complete, ensure the following text is on a new line
    if bar_progress == 100:
        stdout.write('\n')
    stdout.flush()


def average_dicts_values_by_key(dicts):
    """
    Mean average values across multiple dicts with the same key

    :param dicts: a list of dictionaries
    :returns: a dictionary of averaged values
    """
    from collections import Counter

    sums = Counter()
    counters = Counter()
    for itemset in dicts:
        sums.update(itemset)
        counters.update(itemset.keys())

    return {x: float(sums[x])/counters[x] for x in sums.keys()}


def average_median_dicts_values_by_key(dicts):
    """
    Median average values across multiple dicts with the same key

    :param dicts: a list of dictionaries
    :returns: a dictionary of averaged values
    """
    from numpy import median
    from collections import defaultdict

    values = defaultdict(list)

    # Assemble a dict containing all values
    for itemset in dicts:
        for key, value in itemset.items():
            values[key].append(value)
            
    return {key: median(value) for key, value in values.items()}
    

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    import numpy as np
    
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh