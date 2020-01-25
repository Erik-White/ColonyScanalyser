import pytest

from colonyscanalyser.plate import (
    Plate
    )

invalid = [-1, -1.1, "one", None]


@pytest.fixture(params = [1, 2, 123456789])
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
    for i in range(1, 10):
        colonies.append(Colony(i))

    yield colonies

@pytest.fixture(scope="function")
def colony_rand_id(request, colonies):
    from random import randint

    yield randint(1, len(colonies))


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
            assert len([*plate.__iter__()]) == 7

        @pytest.mark.parametrize(
            "colonies, edge_cut, name",
            [
                (list(), 0, ""),
                (list(), 1.0, "Test name"),
                (list(), -1, "1")
            ]
        )
        def test_properties(self, plate, colonies, edge_cut, name):

            plate.colonies = colonies
            plate.edge_cut = edge_cut
            plate.name = name

            assert plate.colonies == colonies
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
        def test_add_colony(self, plate):
            colony = Colony(1)
            plate.add_colony(colony)

            assert plate.colony_count == 1
            assert any(colony.id == item.id for item in plate.colonies)
            with pytest.raises(ValueError):
                plate.add_colony(colony)

        def test_colony_exists(self, plate):
            colony = Colony(1)
            plate.colonies = [colony]

            assert plate.colony_exists(colony)

        def test_colony_id_exists(self, plate):
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

        def test_get_colony(self, plate, colonies, colony_rand_id):
            plate.colonies = colonies

            colony = plate.get_colony(colony_rand_id)

            assert colony is not None
            assert colony.id == colony_rand_id

        def test_remove_colony(self, plate, colonies, colony_rand_id):
            plate.colonies = colonies

            colony = plate.get_colony(colony_rand_id)
            assert colony is not None

            plate.remove_colony(colony_rand_id)
            colony = plate.get_colony(colony_rand_id)
            assert colony is None

        def test_remove_colony_error(self, plate, colonies):
            plate.colonies = colonies

            with pytest.raises(KeyError):
                plate.remove_colony(-1)