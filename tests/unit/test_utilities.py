import pytest
import numpy as np

from colonyscanalyser.utilities import (round_tuple_floats,
                                index_number_to_coordinate,
                                coordinate_to_index_number,
                                progress_bar,
                                average_dicts_values_by_key,
                                average_median_dicts_values_by_key,
                                is_outlier
                                )


class TestRoundTupleFloats():
    @pytest.fixture(params=[
        (1.3285, 1.00001),
        (-95840.3567, 0.0),
        (43.94387678, "string.", 2.567)
        ])
    def tuples(self, request):
        yield request.param

    @pytest.fixture(params=[1, 3, 5])
    def rounding(self, request):
        yield request.param

    def float_precision(self, number):
        _, after = str(number).split('.')
        return len(after)

    def test_rounding(self, tuples, rounding):
        result = round_tuple_floats(tuples, rounding)
        
        for value in result:
            assert self.float_precision(value) <= rounding

    def test_value_error(self):
        with pytest.raises(ValueError):
            round_tuple_floats(1)


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


class TestProgressBar():
    def count_lines(self, text):
        return len(text.split('\n'))

    def test_linecount(self, capsys):
        progress_bar(50)
        progress_bar(75.4)
        captured = capsys.readouterr()

        assert self.count_lines(captured.out) == 1

    def test_linecount_finished(self, capsys):
        progress_bar(100)
        captured = capsys.readouterr()
        
        assert self.count_lines(captured.out) == 2
    
    def test_message(self, capsys):
        message = "Test message"
        progress_bar(100, message = message)
        captured = capsys.readouterr()

        assert captured.out[slice(-len(message) - 1, -1, 1)] == message


class TestAverageDictsByKeys():
    @pytest.fixture(params=[[{
            "key1": 5,
            "key2": 1,
            "key3": 0,
            "key4": -1
        },{
            "key1": 10,
            "key2": 2,
            "key3": 0,
            "key4": 1,
            "key5": 100
        }]])
    def dicts(self, request):
        yield request.param

    @pytest.fixture(params=[{
            "key1": 7.5,
            "key2": 1.5,
            "key3": 0,
            "key4": 0,
            "key5": 100
        }])
    def dicts_averaged(self, request):
        yield request.param

    def test_average(self, dicts, dicts_averaged):
        print(average_dicts_values_by_key(dicts))
        assert average_dicts_values_by_key(dicts) == dicts_averaged

    def test_empty_dicts(self):
        assert average_dicts_values_by_key([{},{}]) == {}


class TestAverageMedianDictsByKeys():
    @pytest.fixture(params=[[{
            "key1": 5,
            "key2": 1,
            "key3": 0,
            "key4": -1
        },{
            "key1": 10,
            "key2": 2,
            "key3": 0,
            "key4": 1,
            "key5": 100
        }]])
    def dicts(self, request):
        yield request.param

    @pytest.fixture(params=[{
            "key1": 7.5,
            "key2": 1.5,
            "key3": 0,
            "key4": 0,
            "key5": 100
        }])
    def dicts_averaged(self, request):
        yield request.param

    def test_average(self, dicts, dicts_averaged):
        assert average_median_dicts_values_by_key(dicts) == dicts_averaged

    def test_empty_dicts(self):
        assert average_median_dicts_values_by_key([{},{}]) == {}


class TestIsOutlier():
    @pytest.mark.parametrize("points, expected", [
        ([1, 1, 10, 1], [False, False, True, False]),
        ([-100, 4.5, 10, 0], [True, False, False, False]),
        ])
    def test_return_outlier(self, points, expected):
        assert is_outlier(np.array(points)).all() == np.array(expected).all()

    def test_empty_array(self):
        assert is_outlier(np.array([])).size == 0