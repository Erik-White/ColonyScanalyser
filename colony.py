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

    def __init__(self, id, timepoints = dict()):
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
            return self.timepoints[time_point]
        else:
            raise ValueError("The requested time point does not exist")

    def append_timepoint(time_point, timepointdata):
        if not self.__timepoints.haskey(timepointdata):
            self.timepoints[time_point] = timepointdata
        else:
            raise ValueError("This time point already exists")
        
    def update_timepoint(time_point, timepointdata):
        self.timepoint[time_point] = timepointdata

    def doubling_time():
        x_pts = list(self.timepoints, key=attrgetter('date_time'))
        y_pts = list(self.timepoints, key=attrgetter('area'))
        window = 10
        return [self.__doubling_time(i, x_pts, y_pts, window) for i in xrange(len(x_pts) - window)]

    def remove_timepoint(time_point):
        del self.timepoint[time_point]

    def area_at_timepoint(time_point):
        return self.timepoint[time_point].area

    def center():
        return sum(self.timepoints, key=attrgetter('center')) / len(self.timepoints)

    def center_at_timepoint(time_point):
        return self.timepoint[time_point].center

    def circularity_at_timepoint(time_point):
        return __circularity(self.timepoint[time_point].area, self.timepoint[time_point].perimiter)
            
    def get_areas():
        return list(self.timepoints, key=attrgetter('area'))

    def time_of_appearance():
        return min(self.timepoints, key=attrgetter('date_time'))

    def __circularity(area, perimiter):
        return (4 * math.pi * area) / (perimeter * perimeter)
    
    def __local_doubling_time(index, x_pts, y_pts, window = 10):

        x1 = x_pts[index]
        y1 = y_pts[index]
        x2 = x_pts[index + window]
        y2 = y_pts[index + window]

        return (x2 - x1) * log(2) / log(y2 / y1)

        
def timepoints_from_image(colonies_dict, image, time_point):
    """
    Store individual colony data in a dict of colonies

    :param colonies_dict: a dict of colony objects
    :param image: a segmented and labelled image as a numpy array
    :param time_point: a datetime object corresponding to the image
    :returns: a dictionary of colony objects
    """
    colonies = colonies_dict.copy()

    for rp in regionprops(image):
        # Create a new time point object to store colony data
            timepoint = colony.Colony.Timepoint(
                date_time = time_point,
                area = rp.area,
                center = rp.centroid,
                diameter = rp.equivalent_diameter,
                perimeter = rp.perimeter
            )
        # Append the data at the time point for each colony
        colonies[rp.label].append_timepoint(timepoint)
    
    return colonies