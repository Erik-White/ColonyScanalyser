from math import pi


class Shape:
    """
    An abstract class to provide the fundamental properties of a surface
    """
    @property
    def area(self):
        raise NotImplementedError("This property must be implemented in a derived class")

    @property
    def center(self):
        try:
            return self.__center
        except AttributeError:
            return None

    @center.setter
    def center(self, val: tuple):
        self.__center = val

    @property
    def depth(self):
        try:
            return self.__depth
        except AttributeError:
            return 0

    @depth.setter
    def depth(self, val: float):
        self.__depth = val

    @property
    def height(self):
        try:
            return self.__height
        except AttributeError:
            return 0

    @height.setter
    def height(self, val: float):
        self.__height = val

    @property
    def perimeter(self):
        raise NotImplementedError("This property must be implemented in a derived class")

    @property
    def width(self):
        try:
            return self.__width
        except AttributeError:
            return 0

    @width.setter
    def width(self, val: float):
        self.__width = val


class Circle(Shape):
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
        return self.perimeter

    @property
    def diameter(self):
        return self.__diameter

    @diameter.setter
    def diameter(self, val: float):
        if val < 0:
            raise ValueError("The diameter must be a number greater than zero")

        self.__diameter = val

    @property
    def height(self):
        return self.diameter

    @property
    def perimeter(self):
        return pi * self.diameter

    @property
    def radius(self):
        return self.diameter / 2

    @property
    def width(self):
        return self.diameter