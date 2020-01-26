import pytest
from datetime import datetime

from colonyscanalyser.base import (
    Identified,
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