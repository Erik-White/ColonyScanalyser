from typing import Union, Tuple
from abc import ABC, abstractmethod
from math import pi


class Shape(ABC):
    """
    An abstract class to provide the fundamental properties of a surface
    """
    @property
    @abstractmethod
    def area(self) -> float:
        raise NotImplementedError("This property must be implemented in a derived class")

    @property
    def center(self) -> Union[Tuple[float, float], Tuple[float, float, float]]:
        try:
            return self.__center
        except AttributeError:
            return None

    @center.setter
    def center(self, val: Union[Tuple[float, float], Tuple[float, float, float]]):
        self.__center = val

    @property
    def depth(self) -> float:
        try:
            return self.__depth
        except AttributeError:
            return 0

    @depth.setter
    def depth(self, val: float):
        self.__depth = val

    @property
    def height(self) -> float:
        try:
            return self.__height
        except AttributeError:
            return 0

    @height.setter
    def height(self, val: float):
        self.__height = val

    @property
    @abstractmethod
    def perimeter(self) -> float:
        raise NotImplementedError("This property must be implemented in a derived class")

    @property
    def width(self) -> float:
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
    def area(self) -> float:
        return pi * self.radius * self.radius

    @property
    def circumference(self) -> float:
        return self.perimeter

    @property
    def diameter(self) -> float:
        return self.__diameter

    @diameter.setter
    def diameter(self, val: float):
        if val < 0:
            raise ValueError("The diameter must be a number greater than zero")

        self.__diameter = val

    @property
    def height(self) -> float:
        return self.diameter

    @property
    def perimeter(self) -> float:
        return pi * self.diameter

    @property
    def radius(self) -> float:
        return self.diameter / 2

    @property
    def width(self) -> float:
        return self.diameter


def circularity(area: float, perimeter: float) -> float:
    """
    Calculate how closely the shape of an object approaches that of a mathematically perfect circle

    A mathematically perfect circle has a circularity of 1

    :param area: the size of the region enclosed by the perimeter
    :param perimeter: the total distance along the edge of a shape
    :returns: a ratio of area to perimiter as a float
    """
    return (4 * pi * area) / (perimeter * perimeter)