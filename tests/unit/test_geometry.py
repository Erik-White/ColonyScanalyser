import pytest

from math import pi
from colonyscanalyser.geometry import (
    Circle
    )


class TestCircle():
    @pytest.fixture(params = [0, 0.1, 1, 1.5, 1000])
    def diameter(self, request):
        yield request.param

    @pytest.fixture(params = [-1, "1", "one"])
    def diameter_invalid(self, request):
        yield request.param

    def test_area(self, diameter):
        circle = Circle(diameter)
        radius = diameter / 2

        assert circle.area == pi * radius * radius

    def test_circumference(self, diameter):
        circle = Circle(diameter)

        assert circle.circumference == pi * diameter

    def test_diameter(self, diameter):
        circle = Circle(diameter)

        assert circle.diameter == diameter

    def test_radius(self, diameter):
        circle = Circle(diameter)

        assert circle.radius == diameter / 2

    def test_invalid(self, diameter_invalid):
        with pytest.raises((TypeError, ValueError)):
            Circle(diameter_invalid)