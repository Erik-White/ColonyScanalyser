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
            diameter = i * 1.0,
            perimeter = i * 1.0,
            color_average = (0, 0, 0)
        ))

    yield timepoints


@pytest.fixture(params = [[0, 0.5, 1, 2, 3]])
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
    def timepoints_iter(self, request, timepoints):
        if request.param == list:
            yield timepoints
        elif request.param == dict:
            yield {t.date_time: t for t in timepoints}
        else:
            yield request.param

    @pytest.fixture
    def colony(self, timepoints):
        yield Colony(1, timepoints)

    @pytest.fixture
    def timepoint_empty(self):
        yield Colony.Timepoint(datetime.now(), 0, 0, (0, 0), 0, 0, (0, 0, 0))

    class TestInitialize():
        @pytest.mark.parametrize("timepoints_iter", [list, dict], indirect = True)
        def test_init(self, timepoints_iter):
            colony = Colony(1, timepoints_iter)

            assert isinstance(colony.timepoints, dict)
            assert len(colony.timepoints) == len(timepoints_iter)

        @pytest.mark.parametrize("timepoints_iter", [int, str], indirect = True)
        def test_iterable(self, timepoints_iter):
            with pytest.raises(ValueError):
                Colony(1, timepoints_iter)

        def test_empty(self):
            colony = Colony(1)

            assert colony.id == 1
            with pytest.raises(ValueError):
                colony.timepoints

    class TestProperties():
        @pytest.mark.parametrize("timepoints_iter", [list, None], indirect = True)
        @pytest.mark.parametrize("id", [-1, 0, 2, 3.4, 1000000, "1"])
        def test_id(self, timepoints_iter, id):
            colony = Colony(id, timepoints_iter)

            assert colony.id == id

        def test_iterable(self, colony):
            assert len([*colony.__iter__()]) == 18

        def test_timepoints(self, timepoints, colony):
            assert len(colony.timepoints) == len(timepoints)
            assert colony.timepoint_first == timepoints[0]
            assert colony.timepoint_last == timepoints[-1]

        def test_center(self, timepoints):
            from statistics import mean

            colony = Colony(id, timepoints)

            for i, coord in enumerate(colony.center):
                assert round(coord, 4) == round(mean([t.center[i] for t in timepoints]), 4)

        def test_growth_rate(self, timepoints, timepoint_empty):
            colony = Colony(1, timepoints)
            colony_empty = Colony(1, [timepoint_empty])

            assert colony.growth_rate == len(timepoints) - 1
            assert colony_empty.growth_rate == 0

        def test_growth_rate_average(self, timepoints, timepoint_empty):
            colony = Colony(id, timepoints)
            colony_empty = Colony(1, [timepoint_empty])

            assert colony.growth_rate_average == ((timepoints[-1].area - timepoints[0].area) ** (1 / len(timepoints))) - 1
            assert colony_empty.growth_rate_average == 0

    class TestTimepoint():
        def test_iterable(self, timepoints):
            from dataclasses import fields

            assert len([*timepoints[0].__iter__()]) == 7
            for value, field in zip([*timepoints[0].__iter__()], fields(Colony.Timepoint)):
                assert isinstance(value, field.type)

    class TestMethods():
        def test_get_timepoint(self, timepoints):
            colony = Colony(1, timepoints)

            # Get timepoint
            assert colony.get_timepoint(timepoints[0].date_time) == timepoints[0]
            with pytest.raises(ValueError):
                colony.get_timepoint(None)

        def test_append_timepoint(self, timepoints, timepoint_empty):
            colony = Colony(1, timepoints)
            colony.append_timepoint(timepoint_empty)

            assert timepoint_empty.date_time in colony.timepoints
            with pytest.raises(ValueError):
                colony.append_timepoint(timepoints[0])

        def test_update_timepoint(self, timepoints, timepoint_empty):
            colony = Colony(1, timepoints)

            colony.update_timepoint(timepoints[0], timepoint_empty)
            assert colony.timepoint_first == timepoint_empty

        def test_remove_timepoint(self, timepoints,):
            colony = Colony(1, timepoints)
            colony.remove_timepoint(timepoints[0].date_time)

            assert timepoints[0].date_time not in colony.timepoints

        @pytest.mark.parametrize("timepoint_index, expected", [(0, 12.57), (-1, 1.4)])
        def test_circularity(self, timepoints, timepoint_index, expected):
            colony = Colony(1, timepoints)
            circularity = colony.circularity_at_timepoint(timepoints[timepoint_index].date_time)

            assert round(circularity, 2) == expected

        @pytest.mark.parametrize("timepoints_iter", [list, None], indirect = True)
        @pytest.mark.parametrize("elapsed_minutes", [True, False])
        @pytest.mark.parametrize(
            "window, expected, expected_minutes",
            [
                (1, timedelta(seconds = 5, microseconds = 884948), 5.88),
                (3, timedelta(seconds = 5, microseconds = 128535), 5.13),
                (5, timedelta(seconds = 4, microseconds = 273778), 4.27)
            ])
        def test_doubling_time(self, timepoints_iter, timepoint_empty, window, elapsed_minutes, expected, expected_minutes):
            colony = Colony(1, timepoints_iter)
            colony_empty = Colony(1, [timepoint_empty])

            if timepoints_iter is not None:
                doubling_times = colony.get_doubling_times(window = window, elapsed_minutes = elapsed_minutes)

                if elapsed_minutes:
                    doubling_times = [round(t, 2) for t in doubling_times]
                    expected = expected_minutes

                assert max(doubling_times) == expected
                assert colony_empty.get_doubling_times() == list()

            else:
                with pytest.raises(ValueError):
                    colony.get_doubling_times(window = window, elapsed_minutes = elapsed_minutes)

        @pytest.mark.parametrize("timepoints_iter", [list, None], indirect = True)
        @pytest.mark.parametrize("elapsed_minutes", [True, False])
        @pytest.mark.parametrize(
            "window, expected, expected_minutes",
            [
                (1, timedelta(seconds = 3, microseconds = 449924), 3.45),
                (3, timedelta(seconds = 3, microseconds = 339683), 3.34),
                (5, timedelta(seconds = 3, microseconds = 126998), 3.13)
            ])
        def test_doubling_time_average(self, timepoints_iter, timepoint_empty, window, elapsed_minutes, expected, expected_minutes):
            colony = Colony(1, timepoints_iter)
            colony_empty = Colony(1, [timepoint_empty])

            if timepoints_iter is not None:
                doubling_time_average = colony.get_doubling_time_average(window = window, elapsed_minutes = elapsed_minutes)

                if elapsed_minutes:
                    doubling_time_average = round(doubling_time_average, 2)
                    expected = expected_minutes

                assert doubling_time_average == expected
                assert colony_empty.get_doubling_time_average() == 0

            else:
                with pytest.raises(ValueError):
                    colony.get_doubling_time_average(window = window, elapsed_minutes = elapsed_minutes)

        def test_local_doubling_time(self, timepoint_empty):
            colony = Colony(1, [timepoint_empty])

            assert colony._Colony__local_doubling_time(0, [0], [0], window = 0) == 0


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