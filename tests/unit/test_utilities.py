import pytest
import numpy as np

from scanlag.utilities import (index_number_to_coordinate,
                                coordinate_to_index_number,
                                average_dicts_values_by_key,
                                is_outlier
                                )

class TestIndexNumberToCoordinate():
    @pytest.fixture(params=[1, 2, 4, 10])
    def index_number(self, request):
        yield request.param

    @pytest.fixture(params=[-1, 0])
    def index_number_invalid(self, request):
        yield request.param

    @pytest.fixture(params=[(1, 1), (10, 1), (3, 2), (5, 5)])
    def lattice(self, request):
        yield request.param

    @pytest.fixture(params=[(0, 0), (-1, 1), (0, 1)])
    def lattice_invalid(self, request):
        yield request.param
        
    @pytest.mark.parametrize("index, lattice, expected", [
        (3, (3, 2), (2, 1)),
        (5, (1, 8), (1, 5)),
        (10, (5, 5), (2, 5)),
        ])
    def test_index_valid(self, index, lattice, expected):
        result = index_number_to_coordinate(index, lattice)
        assert result == expected

    def test_index_invalid(self, index_number_invalid, lattice):
        with pytest.raises(ValueError):
            index_number_to_coordinate(index_number_invalid, lattice)

    def test_lattice_invalid(self, index_number, lattice_invalid):
        with pytest.raises(ValueError):
            index_number_to_coordinate(index_number, lattice_invalid)

    def test_index_error(self, lattice):
        with pytest.raises(IndexError):
            index_number_to_coordinate(100, lattice)


class TestCoordinateToIndexNumber():
    @pytest.mark.parametrize("coordinate, expected", [
        ((3, 2), 6),
        ((1, 8), 8),
        ((5, 5), 25),
        ])
    def test_index_valid(self, coordinate, expected):
        result = coordinate_to_index_number(coordinate)
        assert result == expected


class TestAverageDictsByKeys():
    dicts = [[
        {"key1": 5,
            "key2": 1,
            "key3": 0,
            "key4": -1
        },
        {"key1": 10,
            "key2": 2,
            "key3": 0,
            "key4": 1,
            "key5": 100
        }]]

    dicts_averaged = {
            "key1": 7.5,
            "key2": 1.5,
            "key3": 0,
            "key4": 0,
            "key5": 100
        }

    @pytest.mark.parametrize("dicts", dicts)
    def test_average(self, dicts):
        assert average_dicts_values_by_key(dicts) == self.dicts_averaged

    def test_empty_dicts(self):
        print(average_dicts_values_by_key([{},{}]))
        assert average_dicts_values_by_key([{},{}]) == {}

class TestIsOutlier():
    @pytest.mark.parametrize("points, expected", [
        ([1, 1, 10, 1], [False, False, True, False]),
        ([-100, 4.5, 10, 0], [True, False, False, False]),
        ])
    def test_return_outlier(self, points, expected):
        assert is_outlier(np.array(points)).all() == np.array(expected).all()

    def test_empty_array(self):
        assert is_outlier(np.array([])).size == 0