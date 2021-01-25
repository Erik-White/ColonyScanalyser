import pytest

from colonyscanalyser.utilities import (
    round_tuple_floats,
    progress_bar,
    savgol_filter,
    dicts_merge,
    dicts_mean,
    dicts_median
)


class TestRoundTupleFloats():
    @pytest.fixture(params = [
        (1.3285, 1.00001),
        (-95840.3567, 0.0),
        (43.94387678, "string.", 2.567)
    ])
    def tuples(self, request):
        yield request.param

    @pytest.fixture(params = [1, 3, 5])
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


class TestSavGolFilter():
    @pytest.mark.parametrize("window_length", [3, 4, 5, 7, 10])
    def test_window(self, window_length):
        measurements = [0] * window_length

        results = savgol_filter(measurements, window_length, 1)

        assert (results == measurements).all()

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 6])
    def test_order(self, order):
        measurements = [1] * 10
        window_length = 3

        results = savgol_filter(measurements, window_length, order)

        if order >= window_length:
            assert list(results) == measurements
        else:
            assert (results != measurements).any()


class TestDictsMerge():
    @pytest.mark.parametrize(
        "dicts, expected",
        [
            ([{"one": 1}], {"one": [1]}),
            ([{"one": 1}, {"one": 1, "two": 2}], {"one": [1, 1], "two": [2]}),
            ([{"one": 1, "two": 0.0}, {"three": 1}], {"one": [1], "two": [0.0], "three": [1]}),
        ]
    )
    def test_values(self, dicts, expected):
        assert dicts_merge(dicts) == expected

    @pytest.mark.parametrize(
        "dicts, expected",
        [
            ([{"one": "one"}], {"one": ["one"]}),
            ([{"one": [1, 2]}, {"one": [3], "two": 2}], {"one": [1, 2, 3], "two": [2]}),
            ([{"one": 1, "three": 0.0}, {"three": [1, 2]}, {"three": [[1], 3]}], {"one": [1], "three": [0.0, 1, 2, [1], 3]})
        ]
    )
    def test_iterables(self, dicts, expected):
        assert dicts_merge(dicts) == expected

    def test_empty_dicts(self):
        assert dicts_merge([{}, {}]) == {}


class TestDictsMean():
    @pytest.mark.parametrize(
        "dicts, expected",
        [(
            [
                {"key1": 5, "key2": 1, "key3": 0, "key4": -1},
                {"key1": 10, "key2": 2, "key3": 0, "key4": 1.0, "key5": 100}
            ],
            {"key1": 7.5, "key2": 1.5, "key3": 0, "key4": 0, "key5": 100}
        )]
    )
    def test_average(self, dicts, expected):
        assert dicts_mean(dicts) == expected

    def test_empty_dicts(self):
        assert dicts_mean([{}, {}]) == {}


class TestDictsMedian():
    @pytest.mark.parametrize(
        "dicts, expected",
        [(
            [
                {"key1": 5, "key2": 1, "key3": 0, "key4": -1},
                {"key1": 10, "key2": 2, "key3": 0, "key4": 1.0, "key5": 100}
            ],
            {"key1": 7.5, "key2": 1.5, "key3": 0, "key4": 0, "key5": 100}
        )]
    )
    def test_average(self, dicts, expected):
        assert dicts_median(dicts) == expected

    def test_empty_dicts(self):
        assert dicts_median([{}, {}]) == {}