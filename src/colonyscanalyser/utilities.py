def round_tuple_floats(tuple_item, precision = 2):
    if not isinstance(tuple_item, tuple):
        raise ValueError(f"The object must be of type 'tuple', not type '{type(tuple_item)}'")

    return tuple(map(lambda x: isinstance(x, float) and round(x, precision) or x, tuple_item))


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

    return {x: float(sums[x]) / counters[x] for x in sums.keys()}


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