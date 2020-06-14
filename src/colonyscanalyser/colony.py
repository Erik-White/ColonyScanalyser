from typing import Union, Dict, List, Tuple
from datetime import timedelta
from dataclasses import dataclass
from collections.abc import Collection
from functools import total_ordering
from numpy import ndarray, log2
from colonyscanalyser import config
from .base import Identified, Named
from .utilities import round_tuple_floats
from .imaging import rgb_to_name
from .growth_curve import GrowthCurve


class Colony(Identified, Named, GrowthCurve):
    """
    An object to hold information on a single colony over time
    """
    @dataclass
    @total_ordering
    class Timepoint:
        """
        Colony growth parameters at timed intervals
        """
        timestamp: timedelta
        area: int
        center: tuple
        diameter: float
        perimeter: float
        color_average: tuple

        def __iter__(self):
            return iter([
                self.timestamp.total_seconds() // 60,
                self.area,
                round_tuple_floats(self.center, 2),
                round(self.diameter, 2),
                round(self.perimeter, 2),
                round_tuple_floats(self.color_average, 2),
            ])

        def __eq__(self, other):
            return (self.timestamp == other.timestamp)

        def __ne__(self, other):
            return not (self == other)

        def __lt__(self, other):
            return (self.timestamp < other.timestamp)

    def __init__(self, id: int, timepoints: Collection = None):
        self.id = id
        # Can't set argument default otherwise it is shared across all class instances
        if timepoints is None:
            timepoints = list()
        self.timepoints = timepoints

    def __iter__(self):
        return iter([
            self.id,
            self.time_of_appearance,
            self.time_of_appearance.total_seconds() // 60,
            round_tuple_floats(self.center, 2),
            self.color_name,
            round_tuple_floats(self.color, 2),
            self.growth_curve.lag_time.total_seconds() // 60,
            self.growth_curve.lag_time_std.total_seconds() // 60,
            round(self.growth_curve.growth_rate * 60, 5),
            round(self.growth_curve.growth_rate_std * 60, 7),
            round(self.growth_curve.carrying_capacity, 2),
            round(self.growth_curve.carrying_capacity_std, 4),
            self.growth_curve.doubling_time.total_seconds() // 60,
            self.growth_curve.doubling_time_std.total_seconds() // 60,
            self.timepoint_first.timestamp.total_seconds() // 60,
            round(self.timepoint_first.area, 2),
            round(self.timepoint_first.diameter, 2),
            self.timepoint_last.timestamp.total_seconds() // 60,
            round(self.timepoint_last.area, 2),
            round(self.timepoint_last.diameter, 2)
        ])

    @property
    def center(self) -> Union[Tuple[float, float], Tuple[float, float, float]]:
        centers = [x.center for x in self.timepoints]

        return tuple(sum(x) / len(self.timepoints) for x in zip(*centers))

    @property
    def color(self) -> Tuple[float, float, float]:
        color_averages = [timepoint.color_average for timepoint in self.timepoints]

        return tuple(sum(x) / len(self.timepoints) for x in zip(*color_averages))

    @property
    def color_name(self) -> str:
        return rgb_to_name(self.color, color_spec = "css3")

    @property
    def timepoints(self):
        if len(self.__timepoints) > 0:
            return sorted(self.__timepoints)
        else:
            raise ValueError("No time points are stored for this colony")

    @timepoints.setter
    def timepoints(self, val: Collection):
        if isinstance(val, dict):
            self.__timepoints = list(val.values())
        elif isinstance(val, Collection) and not isinstance(val, str):
            self.__timepoints = [timepoint for timepoint in val]
        else:
            raise ValueError("Timepoints must be supplied as a Dict or other Collection")

    @property
    def timepoint_first(self) -> Timepoint:
        return min(self.timepoints)

    @property
    def timepoint_last(self) -> Timepoint:
        return max(self.timepoints)

    @property
    def time_of_appearance(self) -> timedelta:
        return self.timepoint_first.timestamp

    @property
    def _growth_curve_data(self) -> Dict[timedelta, Union[float, List[float]]]:
        """
        A set of growth measurements over time

        Provides data for growth_curve.fit_curve

        :returns: a dictionary of measurements at time intervals
        """
        return {timepoint.timestamp: log2(timepoint.area) for timepoint in self.timepoints}

    def append_timepoint(self, timepoint: Timepoint):
        """
        Add a Timepoint to the Colony timepoints collection

        :param timepoint: a Timepoint object
        """
        if timepoint not in self.timepoints:
            self.__timepoints.append(timepoint)
        else:
            raise ValueError(f"This time point at {timepoint.timestamp}  already exists")

    def get_timepoint(self, timestamp: timedelta) -> Timepoint:
        """
        Returns a Timepoint object from the Colony timepoints collection

        :param timestamp: the timedelta key for specific Timepoint in the Colony timepoints collection
        :returns: a Timepoint object from the Colony timepoints collection
        """
        return next((timepoint for timepoint in self.timepoints if timepoint.timestamp == timestamp), None)

    def remove_timepoint(self, timestamp: timedelta):
        """
        Remove a specified Timepoint from the Colony timepoints collection

        :param timestamp: the timedelta key for specific Timepoint in the Colony timepoints collection
        """
        timepoint = self.get_timepoint(timestamp)
        del self.__timepoints[self.__timepoints.index(timepoint)]


def timepoints_from_image(
    image_segmented: ndarray,
    timestamp: timedelta,
    image: ndarray = None
) -> List[Colony.Timepoint]:
    """
    Create Timepoint objects from a segemented image

    Optionally include the original un-segmented image to provide colony colour information

    :param image_segmented: a segmented and labelled image as a numpy array
    :param timestamp: a timedelta object corresponding to the image
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
            limit = 200 if image_circle.shape[2] > 3 else 0
            image_circle = image_circle[image_circle[:, :, -1] > limit]
            if image_circle.size > 0:
                # Calculate the average colour values by column over the colony area and remove alpha channel (if present)
                color_average = tuple(image_circle.mean(axis = 0)[:3])

        # Create a new time point object to store colony data
        timepoint_data = Colony.Timepoint(
            timestamp = timestamp,
            area = rp.area,
            center = rp.centroid,
            diameter = rp.equivalent_diameter,
            perimeter = rp.perimeter,
            color_average = color_average
        )

        colonies.append(timepoint_data)

    return colonies


def colonies_filtered(colonies: List[Colony], timestamp_diff_std: float = 10) -> List[Colony]:
    """
    Filter colonies to return only valid colonies

    :param colonies: a list of Colony instances
    :param timestamp_diff_std: the maximum allowed deviation in timestamps (i.e. likelihood of missing data)
    :returns: a filtered list of Colony instances
    """
    from numpy import diff

    # If no objects are found
    if not len(colonies) > 0:
        return colonies

    # Filter colonies to remove noise, background objects and merged colonies
    colonies = list(filter(
        lambda colony:
            # Remove objects that do not have sufficient data points
            len(colony.timepoints) > config.COLONY_TIMEPOINTS_MIN and
            # No colonies should be visible at the start of the experiment
            colony.time_of_appearance.total_seconds() > 0 and
            # Remove objects with large gaps in the data
            diff([t.timestamp.total_seconds() for t in colony.timepoints[1:]]).std() < timestamp_diff_std and
            # Remove object that do not show growth, these are not colonies
            colony.timepoint_last.area > config.COLONY_GROWTH_FACTOR_MIN * colony.timepoint_first.area and
            # Objects that appear with a large initial area are either merged colonies or noise
            colony.timepoint_first.area < config.COLONY_FIRST_AREA_MAX,
            colonies
    ))

    return colonies


def colonies_from_timepoints(
    timepoints: List[Colony.Timepoint],
    distance_tolerance: float = 1
) -> List[Colony]:
    """
    Create a dictionary of Colony objects from Timepoint data

    :param timepoints: a list of Timepoint data objects
    :param distance_tolerance: the difference allowed between centre values
    :returns: a list of Colony objects containing Timepoint data
    """
    colonies = list()

    if not len(timepoints) > 0:
        raise ValueError("No timepoints were supplied")

    # Group Timepoints by centre distances
    colony_centers = group_timepoints_by_center(
        timepoints,
        max_distance = distance_tolerance
    )

    # Create a colony object for each group of centres
    for i, timepoint_objects in enumerate(colony_centers, start = 1):
        # Create a Dict of timepoints with timestamp as the keys
        timepoints_dict = {timepoint.timestamp: timepoint for timepoint in timepoint_objects}
        # Create the Colony object with the Timepoints
        colonies.append(Colony(i, timepoints_dict))

    return colonies


def group_timepoints_by_center(
    timepoints: List[Colony.Timepoint],
    max_distance: float = 1
) -> List[List[Colony.Timepoint]]:
    """
    Split a list of Timepoint objects into sub groups

    Timepoints are grouped by euclidean distance

    :param timepoints: a list of Timepoint objects
    :param max_distance: the difference allowed between centers
    :returns: a list of lists of Timepoint objects
    """
    try:
        from math import dist
    except ImportError:    # pragma: no cover
        # math.dist is not available in Python <3.8
        from scipy.spatial.distance import euclidean as dist

    center_groups = list()
    timepoints = timepoints.copy()

    while len(timepoints) > 0:
        centers = list()

        # Compare current center with remaining centers in the list
        for j, timepoint_compare in reversed(list(enumerate(timepoints))):
            if dist(timepoints[0].center, timepoint_compare.center) <= max_distance:
                # Remove the Timepoint to a group if within distance limit
                centers.append(timepoints.pop(j))

        if centers:
            center_groups.append(centers)

    return center_groups