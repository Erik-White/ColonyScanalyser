import pytest

from colonyscanalyser.plate import (
    Plate
    )

invalid = [-1, -1.1, "one", None]

@pytest.fixture(params = [0, 1, 123456789])
def id(request):
    yield request.param


@pytest.fixture(params = [0, 1.0, 1 * 10**14])
def diameter(request):
    yield request.param


@pytest.fixture(scope="function")
def plate(request, id, diameter):
    yield Plate(id, diameter)


@pytest.fixture(scope="function")
def colonies(request):
    colonies = list()
    for i in range(10):
        colonies.append(Colony(i))

    yield colonies


class Colony():
    def __init__(self, id):
        self.id = id


class TestPlate():
    class TestInitialize():
        def test_init(self, id, diameter):
            plate = Plate(id, diameter)

            assert plate.id == id
            assert plate.diameter == diameter
            assert isinstance(plate.colonies, list)
            assert plate.colony_count == 0

        @pytest.mark.parametrize("id", invalid)
        @pytest.mark.parametrize("diameter", invalid)
        def test_init_invalid(self, id, diameter):
            with pytest.raises((TypeError, ValueError)):
                Plate(id, diameter)

    class TestProperties():
        def test_iterable(self, plate):
            assert len([*plate.__iter__()]) == 10

        @pytest.mark.parametrize(
            "colonies, center, description, edge_cut, name",
            [
                (list(), (0, 0), "", 0, ""),
                (list(), (1, 1.1), "Test description", 1.0, "Test name"),
                (list(), (0, -1), "1", -1, "1")
            ]
        )
        def test_properties(self, plate, colonies, center, description, edge_cut, name):

            plate.colonies = colonies
            plate.center = center
            plate.description = description
            plate.edge_cut = edge_cut
            plate.name = name

            assert plate.colonies == colonies
            assert plate.center == center
            assert plate.description == description
            assert plate.edge_cut == edge_cut
            assert plate.name == name

        @pytest.mark.parametrize("colonies", [[1], {1: 1}])
        def test_colonies_iterable(self, plate, colonies):
            plate.colonies = colonies

            assert isinstance(plate.colonies, list)
            assert len(plate.colonies) == len(colonies)

        @pytest.mark.parametrize("colonies", [1, "1"])
        def test_colonies_iterable_invalid(self, plate, colonies):
            with pytest.raises((TypeError, ValueError)):
                plate.colonies = colonies

    class TestMethods():
        def test_append_colony(self, plate):
            colony = Colony(1)
            plate.append_colony(colony)

            assert plate.colony_count == 1
            assert any(colony.id == item.id for item in plate.colonies)
            with pytest.raises(ValueError):
                plate.append_colony(colony)

        def test_colony_exists(self, plate):
            colony = Colony(1)
            plate.colonies = [colony]

            assert plate.colony_exists(colony)

        def test_colonies_rename_sequential(self, plate, colonies):
            seq_start = 11
            plate.colonies = colonies
            plate.colonies_rename_sequential(start = seq_start)

            assert plate.colony_count == len(colonies)
            for i in range(seq_start, seq_start + len(colonies)):
                assert any(colony.id == i for colony in plate.colonies)