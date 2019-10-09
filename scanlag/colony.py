"""
An object to hold information on a single colony over time
"""
import datetime
from math import pi, log
from dataclasses import dataclass

class Colony():
    @dataclass
    class Timepoint:
        date_time: datetime.datetime
        elapsed_minutes: int
        area: int
        center: tuple
        diameter: float
        perimeter: float

    def __init__(self, id, timepoints = None):
        self.id = id
        # Can't set argument default otherwise it is shared across all class instances
        if timepoints is None:
            timepoints = dict()
        self.timepoints = timepoints

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
        return sum([value.center for key, value in self.timepoints.items()]) / len(self.timepoints)

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
            raise ValueError("The requested time point does not exist")

    def append_timepoint(self, time_point, timepointdata):
        if time_point not in self.__timepoints:
            self.__timepoints[time_point] = timepointdata
        else:
            raise ValueError("This time point already exists")
        
    def update_timepoint(self, time_point, timepointdata):
        self.timepoints[time_point] = timepointdata

    def get_doubling_times(self, window = 10, elapsed_minutes = False):
        if elapsed_minutes:
            x_pts = [value.elapsed_minutes for key, value in self.timepoints.items()]
        else:
            x_pts = [value.date_time for key, value in self.timepoints.items()]
        y_pts = [value.area for key, value in self.timepoints.items()]

        return [self.__local_doubling_time(i, x_pts, y_pts, window) for i in range(len(x_pts) - window)]

    def get_doubling_time_average(self, window = 10, elapsed_minutes = False):
        doubling_times = self.get_doubling_times(window, elapsed_minutes)
        
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

        return (x2 - x1) * log(2) / log(y2 / y1)

        
def timepoints_from_image(colonies_dict, image, time_point, elapsed_minutes):
    """
    Store individual colony data in a dict of colonies

    :param colonies_dict: a dict of colony objects
    :param image: a segmented and labelled image as a numpy array
    :param time_point: a datetime object corresponding to the image
    :param elapsed_minutes: an integer representing the number of minutes since starting
    :returns: a dictionary of colony objects
    """
    from skimage.measure import regionprops
    colonies = colonies_dict.copy()

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
        # Append the data at the time point for each colony
        if rp.label not in colonies:
            colonies[rp.label] = Colony(rp.label)
            
        colonies[rp.label].append_timepoint(timepoint_data.date_time, timepoint_data)
    
    return colonies