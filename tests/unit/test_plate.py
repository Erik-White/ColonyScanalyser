import pytest
from unittest.mock import patch
from datetime import timedelta

from colonyscanalyser.plate import (
    Plate,
    PlateCollection
)

invalid = [-1, -1.1, "one", None]


@pytest.fixture(params = [1, 2, 123456789])
def id(request):
    yield request.param


@pytest.fixture(params = [0, 1.0, 1 * 10**14])
def diameter(request):
    yield request.param


@pytest.fixture(scope = "function")
def plate(request, id, diameter):
    yield Plate(id, diameter)


@pytest.fixture(scope = "function")
def colonies(request):
    colonies = list()
    for i in range(1, 10):
        colonies.append(ColonyMock(i))

    yield colonies


def ColonyMock(id):
    patcher = patch(
        "colonyscanalyser.colony.Colony",
        id = id,
        timepoints = {str(id): str(id)},
        time_of_appearance = timedelta(seconds = id)
    )

    return patcher.start()


class TestPlate():
    class TestInitialize():
        def test_init(self, id, diameter):
            plate = Plate(id, diameter)

            assert plate.id == id
            assert plate.diameter == diameter
            assert isinstance(plate.items, list)
            assert plate.count == 0

        @pytest.mark.parametrize("id", invalid)
        @pytest.mark.parametrize("diameter", invalid)
        def test_init_invalid(self, id, diameter):
            with pytest.raises((TypeError, ValueError)):
                Plate(id, diameter)

    class TestProperties():
        def test_iterable(self, plate, colonies):
            plate.items = colonies

            assert len([*plate.__iter__()]) == 15

        @pytest.mark.parametrize(
            "colonies, edge_cut, name",
            [
                (list(), 0, ""),
                (list(), 1.0, "Test name"),
                (list(), -1, "1")
            ]
        )
        def test_properties(self, plate, colonies, edge_cut, name):

            plate.items = colonies
            plate.edge_cut = edge_cut
            plate.name = name

            assert plate.items == colonies
            assert plate.edge_cut == edge_cut
            assert plate.name == name

    class TestMethods():
        def test_colonies_rename_sequential(self, plate, colonies):
            seq_start = 11
            plate.items = colonies
            plate.colonies_rename_sequential(start = seq_start)

            assert plate.count == len(colonies)
            for i in range(seq_start, seq_start + len(colonies)):
                assert any(colony.id == i for colony in plate.items)

        def test_colonies_to_csv(self, plate, colonies, tmp_path):
            import csv

            plate.items = colonies
            result = plate.colonies_to_csv(tmp_path)

            # Check all rows were written correctly
            with open(result, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for i, row in enumerate(reader):
                    # Skip headers row
                    if i != 0:
                        assert [str(x) for x in colonies[i - 1]] == row

        def test_colonies_timepoints_to_csv(self, plate, colonies, tmp_path):
            import csv

            plate.items = colonies
            result = plate.colonies_timepoints_to_csv(tmp_path)

            # Check all rows were written correctly
            with open(result, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for i, row in enumerate(reader):
                    # Skip headers row
                    if i != 0:
                        assert [str(x) for x in colonies[i - 1].timepoints.items()] == [str(tuple(row))]

        def test_collection_to_csv(self, plate, tmp_path, colonies):
            file_name = "test"
            result = plate._collection_to_csv(str(tmp_path), file_name, colonies, list())

            assert result == tmp_path.joinpath(file_name).with_suffix(".csv")

        def test_collection_to_csv_path(self, plate):
            with pytest.raises(FileNotFoundError):
                plate._collection_to_csv("", "", list())


class TestPlateCollection:
    class TestInitialize:
        @pytest.mark.parametrize("shape", [(3, 1), (5, 5), (2, 2)])
        def test_init(self, shape):
            collection = PlateCollection(shape = shape)

            assert collection.shape == shape

        @pytest.mark.parametrize("shape", [(0, 1), (-1, 1), (1.1, 1)])
        def test_init_invalid(self, shape):
            with pytest.raises(ValueError):
                PlateCollection(shape = shape)

    class TestProperties:
        def test_centers(self):
            centers = list()
            collection = PlateCollection()
            for i in range(1, 10):
                center = (i, i)
                centers.append(center)
                collection.add(id = i, diameter = 1, center = center)

            assert collection.count == len(centers)
            assert collection.centers == centers

    class TestMethods:
        @pytest.fixture
        def image_circle(self, request):
            from numpy import uint8, mgrid

            # Create a 200x200 array with a donut shaped circle around the centre
            xx, yy = mgrid[:200, :200]
            circle = (xx - 100) ** 2 + (yy - 100) ** 2
            img = ((circle < (6400 + 60)) & (circle > (6400 - 60))).astype(uint8)
            img[img == circle] = 255

            yield img

        def test_add(self, id, diameter):
            collection = PlateCollection()
            item_new = collection.add(id = id, diameter = diameter)

            assert collection.count == 1
            assert item_new in collection.items

        def test_from_image(self, image_circle):
            plates = PlateCollection.from_image(
                shape = (1, 1),
                image = image_circle,
                diameter = 180,
            )

            assert plates is not None
            assert isinstance(plates, PlateCollection)

        def test_plates_from_image(self, image_circle):
            label = "label"
            plates = PlateCollection(shape = (1, 1))
            plates.plates_from_image(
                image = image_circle,
                diameter = 180,
                labels = {1: label}
            )

            assert plates.count == 1
            assert plates.centers == [(102, 102)]
            assert plates.items[0].diameter == 160
            assert plates.items[0].name == label

        def test_plates_from_image_invalid(self, image_circle):
            plates = PlateCollection()

            with pytest.raises(ValueError):
                plates.plates_from_image(
                    image = image_circle,
                    diameter = 180
                )

        def test_plates_to_csv(self, image_circle, tmp_path):
            import csv

            plates = PlateCollection.from_image(
                shape = (1, 1),
                image = image_circle,
                diameter = 180,
            )
            result = plates.plates_to_csv(tmp_path)

            # Check all rows were written correctly
            with open(result, 'r') as csvfile:
                reader = list(csv.reader(csvfile))

                assert len(reader) == plates.count + 1
                assert reader[1] == [
                    "1", "", "(102, 102)", "160", "0", "0", "0", "0.0",
                    "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"
                ]

        def test_slice_plate_image(self, image_circle):
            plates = PlateCollection(shape = (1, 1))
            plates.add(
                id = 1,
                diameter = 180,
                edge_cut = 20,
                center = (102, 102)
            )

            images = plates.slice_plate_image(image_circle)

            assert len(images) == 1
            assert images[1].shape == (141, 141)

        @pytest.mark.parametrize(
            "index, shape, expected",
            [
                (3, (3, 2), (2, 1)),
                (5, (1, 8), (1, 5)),
                (10, (5, 5), (2, 5)),
            ])
        def test_index_to_coordinate(self, index, shape, expected):
            result = PlateCollection.index_to_coordinate(index, shape)

            assert result == expected

        @pytest.mark.parametrize(
            "index, shape",
            [
                (-1, (1, 1)),
                (0, (1, 1)),
                (1, (0, 0)),
                (1, (0, 1)),
                (1, (-1, 1)),
            ])
        def test_index_to_coordinate_invalid(self, index, shape):
            with pytest.raises(ValueError):
                PlateCollection.index_to_coordinate(index, shape)

            with pytest.raises(IndexError):
                PlateCollection.index_to_coordinate(100, (1, 1))

        @pytest.mark.parametrize(
            "coordinate, expected",
            [
                ((3, 2), 6),
                ((1, 8), 8),
                ((5, 5), 25),
            ])
        def test_coordinate_to_index(self, coordinate, expected):
            result = PlateCollection.coordinate_to_index(coordinate)

            assert result == expected

        @pytest.mark.parametrize("coordinate", [(0, 0), (-1, 1)])
        def test_coordinate_to_index_invalid(self, coordinate):
            with pytest.raises(ValueError):
                PlateCollection.coordinate_to_index(coordinate)
