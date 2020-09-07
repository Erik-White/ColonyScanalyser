import pytest
from unittest import mock
from datetime import datetime

from colonyscanalyser.base import (
    Identified,
    IdentifiedCollection,
    Named,
    Unique,
    TimeStamped,
    TimeStampElapsed
)


@pytest.fixture(params = [1, 2, 10000000])
def id(request):
    yield request.param


@pytest.fixture(params = [-1, 0, 0.5, "one"])
def id_invalid(request):
    yield request.param


class TestIdentified:
    def test_init(self, id):
        assert Identified(id).id == id

    def test_id(self, id):
        obj = Identified(id)
        obj.id = id * 3

        assert obj.id == id * 3

    def test_id_invalid(self, id_invalid):
        with pytest.raises(ValueError):
            Identified(id_invalid)


class TestIdentifiedCollection:
    @staticmethod
    def IdentifiedMock(id):
        identified = mock.Mock(spec = Identified)
        identified.id = id

        return identified

    @pytest.fixture
    def identified_items(self):
        items = list()

        for i in range(2, 10):
            items.append(self.IdentifiedMock(i))

        return items

    @pytest.fixture(scope = "function")
    def item_rand_id(self, request, identified_items):
        from random import randint

        yield randint(2, len(identified_items))

    class TestInitialize:
        def test_init(self):
            assert IdentifiedCollection().items == list()

        def test_init_list(self, identified_items):
            collection = IdentifiedCollection(identified_items)

            assert collection.items == identified_items

    class TestProperties:
        def test_count(self, identified_items):
            collection = IdentifiedCollection(identified_items)

            assert collection.count == len(identified_items)

        def test_items_none(self):
            collection = IdentifiedCollection(None)
            collection.items = None

            assert collection.items is not None
            assert isinstance(collection.items, list)

        def test_items_sorted(self, identified_items):
            from random import sample

            identified_items_shuffled = sample(identified_items, len(identified_items))
            collection = IdentifiedCollection(identified_items_shuffled)

            assert collection.items != identified_items_shuffled
            assert collection.items == identified_items

        @pytest.mark.parametrize("items", [list(), dict()])
        def test_items_iterable(self, items):
            collection = IdentifiedCollection()
            if isinstance(items, dict):
                items[1] = TestIdentifiedCollection.IdentifiedMock(1)
            else:
                items.append(TestIdentifiedCollection.IdentifiedMock(1))
            collection.items = items

            assert isinstance(collection.items, list)
            assert len(collection.items) == len(items)

        @pytest.mark.parametrize("items", [1, "1"])
        def test_items_iterable_invalid(self, items):
            collection = IdentifiedCollection()

            with pytest.raises((TypeError, ValueError)):
                collection.items = items

    class TestMethods:
        def test_add_item(self, identified_items):
            collection = IdentifiedCollection(identified_items)
            item_new = collection.add(id = 1)

            assert collection.count == len(identified_items) + 1
            assert item_new in collection
            assert collection.items[0] == item_new

        def test_append_item(self):
            collection = IdentifiedCollection()
            identified_item = TestIdentifiedCollection.IdentifiedMock(1)
            collection.append(identified_item)

            assert collection.count == 1
            assert any(identified_item.id == item.id for item in collection.items)
            with pytest.raises(ValueError):
                collection.append(identified_item)

        def test_exists(self, identified_items):
            collection = IdentifiedCollection(identified_items)

            assert collection.exists(identified_items[0].id)

        def test_id_exists(self, identified_items):
            collection = IdentifiedCollection(identified_items)

            assert collection.exists(identified_items[0].id)

        def test_get_item(self, identified_items, item_rand_id):
            collection = IdentifiedCollection(identified_items)

            item = collection[item_rand_id]

            assert item is not None
            assert item.id == item_rand_id

        def test_remove_item(self, identified_items, item_rand_id):
            collection = IdentifiedCollection(identified_items)

            item = collection[item_rand_id]
            assert item is not None

            del collection[item_rand_id]
            assert item not in collection

        def test_remove_item_invalid(self, identified_items):
            collection = IdentifiedCollection(identified_items)

            with pytest.raises(KeyError):
                del collection[-1]


class TestNamed:
    @pytest.fixture(params = ["name", "name with spaces", 1, 0, -1, 1.1])
    def name(self, request):
        yield request.param

    def test_init(self, name):
        assert Named(name).name == str(name)


class TestUnique:
    @pytest.fixture(scope = "class")
    def unique(self):
        yield Unique()

    def test_init(self, unique):
        assert unique.id == 1

    def test_unique(self, unique):
        assert unique.id == 1
        unique = None
        assert Unique().id != 1

    def test_increment(self, unique):
        id_count = unique.id_count
        items_total = 10
        items = list()
        for i in range(items_total):
            items.append(Unique())

        assert len(items) == items_total
        for i in range(items_total):
            assert items[i].id == i + 1 + id_count
        for item in items:
            assert (item.id != existing_item.id for existing_item in items)

    def test_id(self, id):
        obj = Unique()
        original_id = obj.id
        obj.id = id

        assert obj.id == original_id


class TestTimeStamped:
    def test_init(self):
        timestamp = datetime(1, 1, 1)
        timestamped = TimeStamped(timestamp)

        assert timestamped.timestamp == timestamp

    def test_init_auto(self):

        assert TimeStamped().timestamp is not None


class TestTimeStampElapsed:
    def test_init(self):
        timestamp = datetime(1, 1, 1)
        timestampelapsed = TimeStampElapsed(timestamp, timestamp)

        assert timestampelapsed.timestamp == timestamp
        assert timestampelapsed.timestamp_initial == timestamp

    def test_init_auto(self):

        assert TimeStampElapsed().timestamp is not None
        assert TimeStampElapsed().timestamp_initial is not None

    def test_timestamp_elapsed(self):
        timestamp_inital = datetime(1, 1, 1, 0, 0)
        timestamp = datetime(1, 1, 1, 1, 1)
        timestampelapsed = TimeStampElapsed(timestamp, timestamp_inital)
        timestamp_diff = timestamp - timestamp_inital

        assert timestampelapsed.timestamp_elapsed == timestamp_diff
        assert timestampelapsed.timestamp_elapsed_hours == timestamp_diff.total_seconds() / 3600
        assert timestampelapsed.timestamp_elapsed_minutes == int(timestamp_diff.total_seconds() / 60)