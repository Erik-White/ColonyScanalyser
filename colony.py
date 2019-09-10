"""
An object to hold information on a single colony over time
"""
import datetime
from dataclasses import dataclass#only in python >3.7

class Colony(object):# Must derive from object to be new-style class in Python 2

    @dataclass
    class Timepoint:
        date_time: datetime.datetime
        area: int
        center: tuple
        diameter: float
        perimeter: float

    def TimeOfAppearance:
        return 0

    def __init__(self, id, timepoints = list()):
        self.id = id
        self.timepoints = timepoints

    @property
    def timepoints(self):
        if len(self.__timepoints) >0:
            return sorted(self.__timepoints, key=attrgetter('date_time'))
        else:
            raise ValueError("No time points are stored for this colony")

    @timepoints.setter
    def timepoints(self, val):
        self.__timepoints = val

    @property
    def timepoint(self, time_point):
        if self.__timepoints.haskey(time_point):
            return timepoints[time_point]
        else:
            raise ValueError("The requested time point does not exist")

    def append_timepoint(time_point, timepointdata):
        if not self.__timepoints.haskey(key):
            timepoints[time_point] = TimepointData
        else:
            raise ValueError("This time point already exists")
        
    def update_timepoint(time_point, timepointdata):
        timepoint[time_point] = timepointdata

    def doubling_time():
        x_pts = list(timepoints, key=attrgetter('date_time'))
        y_pts = list(timepoints, key=attrgetter('area'))
        window = 10
        return [doubling_time(i, x_pts, y_pts, window) for i in xrange(len(x_pts) - window)]

    def remove_timepoint(time_point):
        del timepoint[time_point]

    def area_at_timepoint(time_point):
        return timepoint[time_point].area

    def center():
        return sum(timepoints, key=attrgetter('center')) / len(timepoints)

    def center_at_timepoint(time_point):
        return timepoint[time_point].center

    def circularity_at_timepoint(time_point):
        return __circularity(timepoint[time_point].area, timepoint[time_point].perimiter)
            
    def get_areas():
        return list(timepoints, key=attrgetter('area'))

    def time_of_appearance():
        return min(timepoints, key=attrgetter('date_time'))

    def __circularity(area, perimiter)
        return (4 * math.pi * area) / (perimeter * perimeter)
    
    def __local_doubling_time(index, x_pts, y_pts, window = 10):

        x1 = x_pts[index]
        y1 = y_pts[index]
        x2 = x_pts[index + window]
        y2 = y_pts[index + window]

        return (x2 - x1) * log(2) / log(y2 / y1)