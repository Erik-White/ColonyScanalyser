import pytest
from datetime import datetime, timedelta

from colonyscanalyser.colony import (
    Colony,
    timepoints_from_image,
    colonies_from_timepoints,
    group_timepoints_by_center
    )


centers = [
    (3, 3.99),
    (3, 3),
    (2, 3.1),
    (2.49, 3),
    (2.51, 3),
    (2.5, 2.99),
    (4, 3.9),
    (3, 10),
    (0, 4.4)
    ]
distances = [0, 0.5, 1, 2, 3]


@pytest.fixture(params = [centers])
def timepoints(request):
    timepoints = list()
    date_time = datetime.now()

    for i, center in enumerate(centers, start = 1):
        timepoints.append(Colony.Timepoint(
            date_time = date_time + timedelta(seconds = i),
            elapsed_minutes = i,
            area = i,
            center = center,
            diameter = i,
            perimeter = i
        ))

    yield timepoints


@pytest.fixture(params = [distances])
def distance(request):
    yield request.param


class TestTimepointsFromImage():
    from numpy import array

    image = array([
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [2, 2, 0, 0, 1, 0, 0, 1, 0],
        [2, 2, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 3],
        [0, 1, 0, 0, 0, 0, 0, 0, 0]])
    centers_expected = [(3.9, 4.12), (1.5, 0.5), (6.0, 8.0)]

    def test_image(self):
        timepoints = timepoints_from_image(self.image, datetime.now(), 0)
        assert len(timepoints) == 3
        assert [timepoint.center in self.centers_expected for timepoint in timepoints]


class TestColony():
    @pytest.fixture
    def timepoints(self, request, timepoints):
        if request.param == list:
            yield timepoints
        elif request.param == dict:
            yield {t.date_time: t for t in timepoints}
        else:
            yield request.param

    @pytest.mark.parametrize("timepoints", [list, dict, int], indirect = True)
    def test_init(self, timepoints):
        if isinstance(timepoints, list) or isinstance(timepoints, dict):
            colony = Colony(1, timepoints)

            assert isinstance(colony.timepoints, dict)
            assert len(colony.timepoints) == len(timepoints)
        else:
            with pytest.raises(ValueError):
                Colony(1, timepoints)

    @pytest.mark.parametrize("timepoints", [list], indirect = True)
    def test_properties(self, timepoints):
        from statistics import mean

        colony = Colony(1, timepoints)

        assert colony.id == 1
        assert [*colony.timepoint_first.__iter__()] == [*timepoints[0]]
        assert len([*colony.__iter__()]) == 16
        assert len(colony.timepoints) == len(timepoints)
        assert colony.timepoint_first == timepoints[0]
        assert colony.timepoint_last == timepoints[-1]
        for i, coord in enumerate(colony.center):
            assert round(coord, 4) == round(mean([t.center[i] for t in timepoints]), 4)
        assert colony.growth_rate == len(timepoints) - 1

    @pytest.mark.parametrize("timepoints", [list], indirect = True)
    def test_timepoint_methods(self, timepoints):
        colony = Colony(1, timepoints)

        # Get timepoint
        assert colony.get_timepoint(timepoints[0].date_time) == timepoints[0]
        with pytest.raises(ValueError):
            colony.get_timepoint(None)
        # Append timepoint
        timepoint = Colony.Timepoint(datetime.now(), 0, 0, (0, 0), 0, 0)
        colony.append_timepoint(timepoint)
        assert timepoint.date_time in colony.timepoints
        with pytest.raises(ValueError):
            colony.append_timepoint(timepoints[0])
        # Update timepoint
        colony.update_timepoint(timepoints[0], timepoint)
        assert colony.timepoint_first == timepoint
        # Remove timepoint
        colony.remove_timepoint(timepoint.date_time)
        assert timepoint.date_time not in colony.timepoints

    @pytest.mark.parametrize("timepoints", [list], indirect = True)
    @pytest.mark.parametrize("timepoint_index, expected", [(0, 12.57), (-1, 1.4)])
    def test_circularity(self, timepoints, timepoint_index, expected):
        colony = Colony(1, timepoints)
        circularity = colony.circularity_at_timepoint(timepoints[timepoint_index].date_time)

        assert round(circularity, 2) == expected

    @pytest.mark.parametrize("timepoints", [list, None], indirect = True)
    @pytest.mark.parametrize(
        "window, elapsed_minutes, expected, expected_avg",
        [
            (1, True, 5.88, 3.45),
            (3, True, 5.13, 3.34),
            (5, False, timedelta(seconds = 4, microseconds = 273778), timedelta(seconds = 3, microseconds = 126998))
        ])
    def test_doubling_time(self, timepoints, window, elapsed_minutes, expected, expected_avg):
        colony = Colony(1, timepoints)

        if timepoints is not None:
            doubling_times = colony.get_doubling_times(window = window, elapsed_minutes = elapsed_minutes)
            doubling_time_average = colony.get_doubling_time_average(window = window, elapsed_minutes = elapsed_minutes)

            if elapsed_minutes:
                doubling_times = [round(t, 2) for t in doubling_times]
                doubling_time_average = round(doubling_time_average, 2)

            assert max(doubling_times) == expected
            assert doubling_time_average == expected_avg

        else:
            with pytest.raises(ValueError):
                colony.get_doubling_times(window = window, elapsed_minutes = elapsed_minutes)

    def test_zero_division(self):
        colony = Colony(1, [Colony.Timepoint(datetime.now(), 0, 0, (0, 0), 0, 0)])

        assert colony.growth_rate == 0
        assert colony.growth_rate_average == 0
        assert colony.get_doubling_times() == list()
        assert colony.get_doubling_time_average() == 0
        assert colony._Colony__local_doubling_time(0, [0], [0], window = 0) == 0

    def test_empty(self):
        colony = Colony(1)

        with pytest.raises(ValueError):
            colony.timepoints


class TestColoniesFromTimepoints():
    @pytest.fixture(params = [[9, 6, 4, 3, 3]])
    def group_expected(self, request):
        yield request.param

    @pytest.fixture
    def distance_expected(self, distance, group_expected):
        yield list(zip(distance, group_expected))

    def test_distance(self, timepoints, distance_expected):
        for distance, expected in distance_expected:
            result = colonies_from_timepoints(timepoints, distance)

            assert len(result) == expected


class TestGroupTimepointsByCenter():
    @pytest.fixture(params = [[7, 4, 3, 2, 2]])
    def group_expected(self, request):
        yield request.param

    @pytest.fixture
    def distance_expected(self, distance, group_expected):
        yield list(zip(distance, group_expected))

    @pytest.mark.parametrize("axis", [0, 1])
    def test_distance(self, timepoints, axis, distance_expected):
        for distance, expected in distance_expected:
            result = group_timepoints_by_center(timepoints, distance, axis)

            assert len(result) == expected

    @pytest.mark.parametrize("axis", [-1, 2, 0.5, "1"])
    def test_axes(self, timepoints, axis):
        with pytest.raises(ValueError):
            group_timepoints_by_center(timepoints, axis = axis)