import datetime
from math import pi, log
from dataclasses import dataclass
from .utilities import round_tuple_floats

class Colony:
    """
    An object to hold information on a single colony over time
    """
    @dataclass
    class Timepoint:
        date_time: datetime.datetime
        elapsed_minutes: int
        area: int
        center: tuple
        diameter: float
        perimeter: float
        
        def __iter__(self):
            return iter([
                self.date_time,
                self.elapsed_minutes,
                self.area,
                round_tuple_floats(self.center, 2),
                round(self.diameter, 2),
                round(self.perimeter, 2)
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
        self.__timepoints = val

    @property
    def timepoint_first(self):
        return self.get_timepoint(min(self.timepoints.keys()))

    @property
    def timepoint_last(self):
        return self.get_timepoint(max(self.timepoints.keys()))

    @property
    def areas(self):
        return [value.area for key, value in self.timepoints.items()]

    @property
    def center(self):
        centers = [x.center for x in self.timepoints.values()]
        return tuple(sum(x) / len(self.timepoints) for x in zip(*centers))

    @property
    def growth_rate(self):
        return (self.timepoint_last.area - self.timepoint_first.area) / self.timepoint_first.area

    @property
    def growth_rate_average(self):
        if (self.timepoint_last.area - self.timepoint_first.area) / self.timepoint_first.area <= 0:
            return 0
        else:
            return ((self.timepoint_last.area - self.timepoint_first.area) ** (1 / len(self.timepoints))) - 1

    @property
    def time_of_appearance(self):
        return self.timepoint_first.date_time

    def get_timepoint(self, time_point):
        if time_point in self.__timepoints:
            return self.timepoints[time_point]
        else:
            raise ValueError(f"The requested time point ({time_point}) does not exist")

    def append_timepoint(self, time_point, timepointdata):
        if time_point not in self.__timepoints:
            self.__timepoints[time_point] = timepointdata
        else:
            raise ValueError(f"This time point ({time_point})  already exists")
        
    def update_timepoint(self, time_point, timepointdata):
        self.timepoints[time_point] = timepointdata

    def get_doubling_times(self, window = 10, elapsed_minutes = False):
        if elapsed_minutes:
            x_pts = [value.elapsed_minutes for key, value in self.timepoints.items()]
        else:
            x_pts = [value.date_time for key, value in self.timepoints.items()]
        y_pts = [value.area for key, value in self.timepoints.items()]

        if not len(x_pts) > 0 or not len(y_pts) > 0:
            return []

        return [self.__local_doubling_time(i, x_pts, y_pts, window) for i in range(len(x_pts) - window)]

    def get_doubling_time_average(self, window = 10, elapsed_minutes = False):
        doubling_times = self.get_doubling_times(window, elapsed_minutes)
        if not len(doubling_times) > 0:
            return 0

        return sum(doubling_times) / len(doubling_times)

    def remove_timepoint(self, time_point):
        del self.timepoints[time_point]

    def area_at_timepoint(self, time_point):
        return self.get_timepoint(time_point).area

    def center_at_timepoint(self, time_point):
        return self.get_timepoint(time_point).center

    def circularity_at_timepoint(self, time_point):
        return self.__circularity(self.get_timepoint(time_point).area, self.get_timepoint(time_point).perimiter)

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


def timepoints_from_image(image, time_point, elapsed_minutes):
    """
    Create Timepoint objects from a segemented image

    :param image: a segmented and labelled image as a numpy array
    :param time_point: a datetime object corresponding to the image
    :param elapsed_minutes: an integer representing the number of minutes since starting
    :returns: a list of colony objects
    """
    from skimage.measure import regionprops

    colonies = list()

    for rp in regionprops(image, coordinates = "rc"):
        # Create a new time point object to store colony data
        timepoint_data = Colony.Timepoint(
            date_time = time_point,
            elapsed_minutes = elapsed_minutes,
            area = rp.area,
            center = rp.centroid,
            diameter = rp.equivalent_diameter,
            perimeter = rp.perimeter
        )
        
        colonies.append(timepoint_data)
    
    return colonies


def colonies_from_timepoints(timepoints):
    """
    Create a dictionary of Colony objects from Timepoint data

    :param timepoints: a list of Timepoint data objects
    :returns: a list of Colony objects containing Timepoint data
    """
    from collections import defaultdict

    colony_centers = defaultdict(list)
    colonies = list()

    # Build lists of Timepoints, grouped by centres as dict keys
    for timepoint in timepoints:
        colony_centers[round_tuple_floats(timepoint.center, 0)].append(timepoint)
        
    # Create a colony object for each group of centres
    for i, timepoint_objects in enumerate(colony_centers.values(), start = 1):
        # Create a Dict of timepoints with date_time as the keys
        timepoints_dict = {timepoint.date_time : timepoint for timepoint in timepoint_objects}
        # Create the Colony object with the Timepoints
        colonies.append(Colony(i, timepoints_dict))

    return colonies