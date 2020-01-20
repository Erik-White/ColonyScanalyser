import pytest

from math import pi
from colonyscanalyser.geometry import (
    Shape,
    Circle
    )


class TestShape:
    @pytest.fixture
    def shape(self, request):
        yield Shape()

    @pytest.fixture(params = [0, 1, 1.1, -1])
    def distance(self, request):
        yield request.param

    def test_area(self, shape):
        with pytest.raises(NotImplementedError):
            shape.area

    @pytest.mark.parametrize("center", [(0, 0), (1, 1), (0, -1), (0, 0, 0), (1, 1, 1)])
    def test_center(self, shape, center):
        assert shape.center is None

        shape.center = center
        assert shape.center == center

    def test_depth(self, shape, distance):
        assert shape.depth == 0

        shape.depth = distance
        assert shape.depth == distance

    def test_height(self, shape, distance):
        assert shape.height == 0

        shape.height = distance
        assert shape.height == distance

    def test_perimeter(self, shape):
        with pytest.raises(NotImplementedError):
            shape.perimeter

    def test_width(self, shape, distance):
        assert shape.width == 0

        shape.width = distance
        assert shape.width == distance


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

    def test_height(self, diameter):
        circle = Circle(diameter)

        assert circle.height == diameter

    def test_perimeter(self, diameter):
        circle = Circle(diameter)

        assert circle.perimeter == pi * diameter

    def test_radius(self, diameter):
        circle = Circle(diameter)

        assert circle.radius == diameter / 2

    def test_width(self, diameter):
        circle = Circle(diameter)

        assert circle.width == diameter

    def test_diameter_invalid(self, diameter_invalid):
        with pytest.raises((TypeError, ValueError)):
            Circle(diameter_invalid)