from math import pi


class Circle:
    """
    An object to generate the properties of a circle
    """
    def __init__(self, diameter: float):
        self.diameter = diameter

    @property
    def area(self):
        return pi * self.radius * self.radius

    @property
    def circumference(self):
        return pi * self.diameter

    @property
    def diameter(self):
        return self.__diameter

    @diameter.setter
    def diameter(self, val: float):
        if val < 0:
            raise ValueError("The diameter must be a number greater than zero")

        self.__diameter = val

    @property
    def radius(self):
        return self.diameter / 2