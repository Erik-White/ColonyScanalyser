from datetime import datetime, timedelta
from math import pi, log
from dataclasses import dataclass
from collections.abc import Iterable
from .utilities import round_tuple_floats
from.imaging import rgb_to_name


class Colony:
    """
    An object to hold information on a single colony over time
    """
    @dataclass
    class Timepoint:
        date_time: datetime
        elapsed_minutes: int
        area: int
        center: tuple
        diameter: float
        perimeter: float
        color_average: tuple

        def __iter__(self):
            return iter([
                self.date_time,
                self.elapsed_minutes,
                self.area,
                round_tuple_floats(self.center, 2),
                round(self.diameter, 2),
                round(self.perimeter, 2),
                round_tuple_floats(self.color_average, 2),
                ])

    def __init__(self, id, timepoints = None):
        self.id = id
        # Can't set argument default otherwise it is shared across all class instances
        if timepoints is None:
            timepoints = dict()
        self.timepoints = timepoints

    def __iter__(self):
        return iter([
            self.id,
            self.time_of_appearance,
            self.timepoint_first.elapsed_minutes,
            round_tuple_floats(self.center, 2),
            self.color_name,
            round_tuple_floats(self.color, 2),
            round(self.growth_rate_average, 2),
            round(self.growth_rate, 2),
            round(self.get_doubling_time_average(elapsed_minutes = True), 2),
            round_tuple_floats(tuple(self.get_doubling_times(elapsed_minutes = True)), 2),
            self.timepoint_first.elapsed_minutes,
            round_tuple_floats(self.timepoint_first.center, 2),
            self.timepoint_first.area,
            round(self.timepoint_first.diameter, 2),
            self.timepoint_last.elapsed_minutes,
            round_tuple_floats(self.timepoint_last.center, 2),
            self.timepoint_last.area,
            round(self.timepoint_last.diameter, 2)
            ])

    @property
    def timepoints(self):
        if len(self.__timepoints) > 0:
            return self.__timepoints
        else:
            raise ValueError("No time points are stored for this colony")

    @timepoints.setter
    def timepoints(self, val):
        if isinstance(val, dict):
            self.__timepoints = val
        elif isinstance(val, Iterable) and not isinstance(val, str):
            self.__timepoints = {timepoint.date_time: timepoint for timepoint in val}
        else:
            raise ValueError("Timepoints must be supplied as a Dict or other iterable")

    @property
    def timepoint_first(self):
        return self.get_timepoint(min(self.timepoints.keys()))

    @property
    def timepoint_last(self):
        return self.get_timepoint(max(self.timepoints.keys()))

    @property
    def center(self):
        centers = [x.center for x in self.timepoints.values()]
        return tuple(sum(x) / len(self.timepoints) for x in zip(*centers))

    @property
    def color(self):
        color_averages = [x.color_average for x in self.timepoints.values()]
        return tuple(sum(x) / len(self.timepoints) for x in zip(*color_averages))

    @property
    def color_name(self):
        return rgb_to_name(self.color, color_spec = "css3")

    @property
    def growth_rate(self):
        try:
            return (self.timepoint_last.area - self.timepoint_first.area) / self.timepoint_first.area
        except ZeroDivisionError:
            return 0

    @property
    def growth_rate_average(self):
        if self.growth_rate == 0:
            return 0
        else:
            return ((self.timepoint_last.area - self.timepoint_first.area) ** (1 / len(self.timepoints))) - 1

    @property
    def time_of_appearance(self):
        return self.timepoint_first.date_time

    def get_timepoint(self, date_time):
        if date_time in self.__timepoints:
            return self.timepoints[date_time]
        else:
            raise ValueError(f"The requested time point ({date_time}) does not exist")

    def append_timepoint(self, timepoint):
        if timepoint.date_time not in self.__timepoints:
            self.__timepoints[timepoint.date_time] = timepoint
        else:
            raise ValueError(f"This time point ({timepoint.date_time})  already exists")

    def update_timepoint(self, timepoint_original, timepoint_new):
        self.timepoints[timepoint_original.date_time] = timepoint_new

    def remove_timepoint(self, date_time):
        del self.timepoints[date_time]

    def circularity_at_timepoint(self, date_time):
        return self.__circularity(self.get_timepoint(date_time).area, self.get_timepoint(date_time).perimeter)

    def get_doubling_times(self, window = 10, elapsed_minutes = False):
        timepoint_count = len(self.timepoints)
        if timepoint_count <= 1:
            return list()

        if window > timepoint_count:
            window = timepoint_count - 1

        if elapsed_minutes:
            x_pts = [value.elapsed_minutes for key, value in self.timepoints.items()]
        else:
            x_pts = [value.date_time for key, value in self.timepoints.items()]
        y_pts = [value.area for key, value in self.timepoints.items()]

        return [self.__local_doubling_time(i, x_pts, y_pts, window) for i in range(len(x_pts) - window)]

    def get_doubling_time_average(self, window = 10, elapsed_minutes = False):
        doubling_times = self.get_doubling_times(window, elapsed_minutes)

        if not len(doubling_times) > 0:
            return 0

        if elapsed_minutes:
            time_sum = sum(doubling_times)
        else:
            time_sum = sum(doubling_times, timedelta())

        return time_sum / len(doubling_times)

    def __circularity(self, area, perimeter):
        return (4 * pi * area) / (perimeter * perimeter)

    def __local_doubling_time(self, index, x_pts, y_pts, window = 10):

        x1 = x_pts[index]
        y1 = y_pts[index]
        x2 = x_pts[index + window]
        y2 = y_pts[index + window]

        try:
            return (x2 - x1) * log(2) / log(y2 / y1)
        except ZeroDivisionError:
            return 0


def timepoints_from_image(image_segmented, time_point, elapsed_minutes, image = None):
    """
    Create Timepoint objects from a segemented image

    Optionally include the original un-segmented image to provide colony colour information

    :param image_segmented: a segmented and labelled image as a numpy array
    :param time_point: a datetime object corresponding to the image
    :param elapsed_minutes: an integer representing the number of minutes since starting
    :param image: a colour image that image_segmented is derived from
    :returns: a list of colony objects
    """
    from .imaging import cut_image_circle
    from skimage.measure import regionprops

    colonies = list()

    if image is not None:
        if image.shape[:2] != image_segmented.shape[:2]:
            raise ValueError("The image and its segmented image must be the same size")

    for rp in regionprops(image_segmented):
        color_average = (0, 0, 0)
        if image is not None:
            # Select an area of the colony slightly smaller than its full radius
            # This avoids the edge halo of the image which may contain background pixels
            radius = (rp.equivalent_diameter / 2) - ((rp.equivalent_diameter / 2) * 0.10)
            image_circle = cut_image_circle(image[rp.slice], radius - 1)
            # Filter out fringe alpha values and empty pixels
            limit = 0
            if image_circle.shape[2] > 3:
                limit = 200
            image_circle = image_circle[image_circle[:, :, -1] > limit]
            # Calculate the average colour values by column over the colony area and remove alpha channel (if present)
            color_average = tuple(image_circle.mean(axis = 0)[:3])

        # Create a new time point object to store colony data
        timepoint_data = Colony.Timepoint(
            date_time = time_point,
            elapsed_minutes = elapsed_minutes,
            area = rp.area,
            center = rp.centroid,
            diameter = rp.equivalent_diameter,
            perimeter = rp.perimeter,
            color_average = color_average
        )

        colonies.append(timepoint_data)

    return colonies


def colonies_from_timepoints(timepoints, distance_tolerance = 1):
    """
    Create a dictionary of Colony objects from Timepoint data

    :param timepoints: a list of Timepoint data objects
    :param distance_tolerance: the difference allowed between centre values
    :returns: a list of Colony objects containing Timepoint data
    """

    colony_centers = list()
    colonies = list()

    if not len(timepoints) > 0:
        raise ValueError("No timepoints were supplied")

    # First group by row values
    center_groups = group_timepoints_by_center(
        timepoints,
        max_distance = distance_tolerance,
        axis = 0
        )

    # Then split the groups further by column values
    for timepoint_group in center_groups:
        group = group_timepoints_by_center(
            timepoint_group,
            max_distance = distance_tolerance,
            axis = 1
            )
        colony_centers.extend(group)

    # Create a colony object for each group of centres
    for i, timepoint_objects in enumerate(colony_centers, start = 1):
        # Create a Dict of timepoints with date_time as the keys
        timepoints_dict = {timepoint.date_time: timepoint for timepoint in timepoint_objects}
        # Create the Colony object with the Timepoints
        colonies.append(Colony(i, timepoints_dict))

    return colonies


def group_timepoints_by_center(timepoints, max_distance = 1, axis = 0):
    """
    Split a list of Timepoint objects into sub groups
    Compares difference in values along a specified axis

    :param timepoints: a list of Timepoint objects
    :param max_distance: the difference allowed between centre values
    :param axis: the axis index to compare values along
    :returns: a list of lists of Colony objects containing Timepoint objects
    """
    from collections import defaultdict

    # Check that the specified axis exists
    axes = range(len(timepoints[0].center))
    if type(axis) is not int or axis < 0 or axis > max(axes):
        raise ValueError(f"The specified axis ({axis}) is not available. Available axes: {[*axes]}")

    center_groups = defaultdict(list)
    group_count = 0
    timepoint_prev = None

    # Sort the list of timepoints along the specified axis
    for timepoint in sorted(timepoints, key = lambda k: k.center[axis]):
        # Create a new group each time the tolerance limit is reached
        if (timepoint_prev is None or
                abs(timepoint.center[axis] - timepoint_prev.center[axis]) > max_distance):
            group_count += 1
            timepoint_prev = timepoint
        center_groups[group_count].append(timepoint)

    return [list(centers) for centers in center_groups.values()]