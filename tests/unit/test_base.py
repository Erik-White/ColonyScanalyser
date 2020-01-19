import pytest

from colonyscanalyser.base import (
    Identified,
    Named,
    Unique
    )


@pytest.fixture(params = [0, 1, 2, 10000000])
def id(request):
    yield request.param


@pytest.fixture(params = [-1, 0.5, "one"])
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
        assert Named(name).name == name


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