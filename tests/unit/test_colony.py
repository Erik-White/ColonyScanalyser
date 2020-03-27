import pytest
from datetime import timedelta

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
    (0, 4.4),
    (4.2, 3.1),
    (4, 4)
]


@pytest.fixture(params = [centers])
def timepoints(request):
    timepoints = list()

    for i, center in enumerate(centers, start = 1):
        timepoints.append(Colony.Timepoint(
            timestamp = timedelta(seconds = i),
            area = i,
            center = center,
            diameter = i * 1.0,
            perimeter = i * 1.0,
            color_average = (0, 0, 0)
        ))

    yield timepoints


@pytest.fixture(params = [[0, 0.5, 1, 2, 3, 5, 10]])
def distance(request):
    yield request.param


class TestTimepointsFromImage():
    from numpy import array

    @pytest.fixture(params = [array([
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [2, 2, 0, 0, 1, 0, 0, 1, 0],
        [2, 2, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 3],
        [0, 1, 0, 0, 0, 0, 0, 0, 0]])])
    def image(self, request):
        yield request.param

    @pytest.fixture(params = ["rgb", "rgba"])
    def image_rgb(self, request, image):
        from numpy import broadcast_to, newaxis

        array_z = 4
        if request.param == "rgb":
            array_z = 3

        image_rgb = image.copy()
        # Reshape the array to 3 dimensions and fill with the existing values
        image_rgb = broadcast_to(image_rgb[..., newaxis], (image_rgb.shape[0], image_rgb.shape[1], array_z))

        yield image_rgb

    @pytest.fixture(params = [(3.9, 4.12), (1.5, 0.5), (6.0, 8.0)])
    def centers_expected(self, request):
        yield request.param

    def test_image_grayscale(self, image, centers_expected):
        timepoints = timepoints_from_image(image, timedelta.min)
        assert len(timepoints) == 3
        assert [timepoint.center in centers_expected for timepoint in timepoints]

    def test_image_rgb(self, image, image_rgb, centers_expected):
        timepoints = timepoints_from_image(image, timedelta.min, image = image_rgb)
        assert len(timepoints) == 3
        assert [timepoint.center in centers_expected for timepoint in timepoints]

    def test_image_shape(self, image):
        with pytest.raises(ValueError):
            timepoints_from_image(image, timedelta.min, image[:1])


class TestColony():
    @pytest.fixture
    def timepoints_iter(self, request, timepoints):
        if request.param == list:
            yield timepoints
        elif request.param == dict:
            yield {t.timestamp: t for t in timepoints}
        else:
            yield request.param

    @pytest.fixture
    def colony(self, timepoints):
        yield Colony(1, timepoints)

    @pytest.fixture
    def timepoint_empty(self):
        yield Colony.Timepoint(timedelta.min, 0, (0, 0), 0, 0, (0, 0, 0))

    class TestInitialize():
        @pytest.mark.parametrize("timepoints_iter", [list, dict], indirect = True)
        def test_init(self, timepoints_iter):
            colony = Colony(1, timepoints_iter)

            assert isinstance(colony.timepoints, list)
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
        def test_iterable(self, colony):
            assert len([*colony.__iter__()]) == 20

        def test_timepoints(self, timepoints, colony):
            assert len(colony.timepoints) == len(timepoints)
            assert colony.timepoint_first == timepoints[0]
            assert colony.timepoint_last == timepoints[-1]

        def test_center(self, timepoints):
            from statistics import mean

            colony = Colony(1, timepoints)

            for i, coord in enumerate(colony.center):
                assert round(coord, 4) == round(mean([t.center[i] for t in timepoints]), 4)

    class TestTimepoint():
        def test_iterable(self, timepoints):
            from dataclasses import fields

            assert len([*timepoints[0].__iter__()]) == 6
            for value, field in zip([*timepoints[0].__iter__()], fields(Colony.Timepoint)):
                assert isinstance(value, field.type)

    class TestMethods():
        def test_get_timepoint(self, timepoints):
            colony = Colony(1, timepoints)

            assert colony.get_timepoint(timepoints[0].timestamp) == timepoints[0]
            assert colony.get_timepoint(timedelta(seconds = -1)) is None

        def test_append_timepoint(self, timepoints, timepoint_empty):
            colony = Colony(1, timepoints)
            colony.append_timepoint(timepoint_empty)

            assert timepoint_empty in colony.timepoints
            with pytest.raises(ValueError):
                colony.append_timepoint(timepoints[0])

        def test_remove_timepoint(self, timepoints):
            colony = Colony(1, timepoints)
            colony.remove_timepoint(timepoints[0].timestamp)

            assert timepoints[0] not in colony.timepoints


class TestColoniesFromTimepoints():
    @pytest.mark.parametrize("group_expected", [[11, 8, 5, 3, 3, 2, 1]])
    def test_groups(self, timepoints, distance, group_expected):
        for max_dist, expected in zip(distance, group_expected):
            result = colonies_from_timepoints(timepoints, max_dist)

            assert len(result) == expected

    def test_timepoints_empty(self):
        with pytest.raises(ValueError):
            colonies_from_timepoints([])


class TestGroupTimepointsByCenter():
    @pytest.mark.parametrize("group_expected", [[11, 8, 5, 3, 3, 2, 1]])
    def test_distance(self, timepoints, distance, group_expected):
        from math import dist

        for max_dist, expected in zip(distance, group_expected):
            result = group_timepoints_by_center(timepoints, max_dist)

            assert len(result) == expected
            for group in result:
                # Find rough centre point for the group
                x = [timepoint.center[0] for timepoint in group]
                y = [timepoint.center[1] for timepoint in group]
                group_center = (sum(x) / len(group), sum(y) / len(group))

                # Check that points are not too far from centre
                for timepoint in group:
                    assert dist(group_center, timepoint.center) <= max_dist

    def test_timepoints_empty(self):
        assert group_timepoints_by_center([]) == list()