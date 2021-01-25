from typing import Collection, Tuple, List, Dict, Any


def round_tuple_floats(tuple_item: Tuple[float], precision: int = 2) -> Tuple[float]:
    """
    Round float values within a tuple to the required precision

    :param tuple_item: a tuple of floats
    :param precision: the rounding value for the floats in tuple_items
    :returns: a tuple of rounded values
    """
    if not isinstance(tuple_item, tuple):
        raise ValueError(f"The object must be of type 'tuple', not type '{type(tuple_item)}'")

    return tuple(map(lambda x: isinstance(x, float) and round(x, precision) or x, tuple_item))


def savgol_filter(measurements: Collection[Any], window: int = 15, order: int = 2) -> Collection[Any]:
    """
    Smooth a one dimensional set of data with a Savitzky-Golay filter

    If no filtering can be performed the original data is returned unaltered

    :param measurements: a collection of values to filter
    :param window: the window length used in the filter
    :param order: the polynomial order used to fit the values
    :returns: filtered values, if possible
    """
    from scipy.signal import savgol_filter

    # Window length must be odd and greater than polyorder for Savitzky-Golay filter
    if window > len(measurements):
        window = len(measurements)
    if window % 2 == 0:
        window -= 1
    if window >= 1 and window > order:
        measurements = savgol_filter(measurements, window, order)

    return measurements


def progress_bar(bar_progress: float, bar_length: float = 30, message: str = ""):
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


def dicts_merge(dicts: List[dict]):
    """
    Combine values by key from multiple dicts

    :param dicts: a list of dicts to merge
    :returns: a dict containing combined lists of values
    """
    from collections import defaultdict
    from itertools import chain
    from operator import methodcaller

    results = defaultdict(list)

    dict_items = map(methodcaller('items'), dicts)
    for key, value in chain.from_iterable(dict_items):
        if isinstance(value, Collection) and not isinstance(value, str):
            results[key].extend(value)
        else:
            results[key].append(value)

    return results


def dicts_mean(dicts: List[Dict]) -> Dict:
    """
    Mean average values across multiple dicts with the same key

    :param dicts: a list of dictionaries
    :returns: a dictionary of averaged values
    """
    from statistics import mean

    return {key: mean(value) for key, value in dicts_merge(dicts).items()}


def dicts_median(dicts: List[Dict]) -> Dict:
    """
    Median average values across multiple dicts with the same key

    :param dicts: a list of dictionaries
    :returns: a dictionary of averaged values
    """
    from statistics import median

    return {key: median(value) for key, value in dicts_merge(dicts).items()}