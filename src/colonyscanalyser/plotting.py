from typing import Union, Tuple, List
from matplotlib.axes import Axes, BarContainer


def rc_to_xy(coordinate: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert an row, column coordinate to an x, y coordinate

    :param coordinate: an RC coordinate tuple
    :returns: an XY coordinate tuple
    """
    return (coordinate[1], coordinate[0])


def label_bars(ax: Axes, bars: BarContainer, text_format: str, **kwargs):
    """
    Attaches a label on every bar of a regular or horizontal bar chart

    :param ax: a Matplotlib Axes object to modify
    :param bars: a collection of artists for a bar plot
    :param text_format: formatting options for the bar labels
    :param kwargs: arguments to pass through to ax.text
    """
    ys = [bar.get_y() for bar in bars]
    vertical = all(y == ys[0] for y in ys)

    if vertical:
        _label_bar(ax, bars, text_format, **kwargs)
    else:
        _label_bar_horizontal(ax, bars, text_format, **kwargs)


def axis_minutes_to_hours(labels: Union[List[int], List[float]]) -> List[str]:
    """
    Format axis labels in minutes to integer hours

    :param labels: a list of numerical labels to format
    :returns: a list list of labels
    """
    return ["{:.0f}".format(x // 60) for x in labels]


def _label_bar(ax: Axes, bars: BarContainer, text_format: str, **kwargs):
    """
    Attach a text label to each bar displaying its y value

    :param ax: a Matplotlib Axes object to modify
    :param bars: a collection of artists for a bar plot
    :param text_format: formatting options for the bar labels
    :param kwargs: arguments to pass through to ax.text
    """
    max_y_value = ax.get_ylim()[1]
    inside_distance = max_y_value * 0.05
    outside_distance = max_y_value * 0.01

    for bar in bars:
        text = text_format.format(bar.get_height())
        text_x = bar.get_x() + bar.get_width() / 2

        is_inside = bar.get_height() >= max_y_value * 0.15
        if is_inside:
            color = "white"
            text_y = bar.get_height() - inside_distance
        else:
            color = "black"
            text_y = bar.get_height() + outside_distance

        ax.text(text_x, text_y, text, ha = "center", va = "bottom", color = color, **kwargs)


def _label_bar_horizontal(ax: Axes, bars: BarContainer, text_format: str, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    Note: label always outside. otherwise it's too hard to control as numbers can be very long

    :param ax: a Matplotlib Axes object to modify
    :param bars: a collection of artists for a bar plot
    :param text_format: formatting options for the bar labels
    :param kwargs: arguments to pass through to ax.text
    """
    max_x_value = ax.get_xlim()[1]
    distance = max_x_value * 0.0025

    for bar in bars:
        text = text_format.format(bar.get_width())

        text_x = bar.get_width() + distance
        text_y = bar.get_y() + bar.get_height() / 2

        ax.text(text_x, text_y, text, va='center', **kwargs)