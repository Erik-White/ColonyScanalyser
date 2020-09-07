from __future__ import annotations
from typing import Type, Union, Iterable, List
from collections.abc import Collection
from datetime import datetime, timedelta


class Identified:
    """
    An object with a integer ID number
    """
    def __init__(self, id: int):
        self.id = id

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, val: int):
        if self._id_is_valid(val):
            self._id = val
        else:
            raise ValueError(f"'{val}' is not a valid id. An id must be a non-negative integer'")

    @staticmethod
    def _id_is_valid(id: int) -> bool:
        """
        Verifies if a value conforms to the requirements for an ID number

        An ID number is an integer with a value greater than zero

        :param id: an ID number to verify
        :returns: True if the value conforms to the requirements for an ID number
        """
        return isinstance(id, int) and id > 0


class IdentifiedCollection(Collection):
    """
    An collection of Identified objects with generic methods for modifying the collection
    """
    def __init__(self, items: Collection = None):
        self.items = items

    @property
    def count(self) -> int:
        return self.__len__()

    @property
    def items(self) -> List[Identified]:
        """
        Returns a sorted list of items from the collection

        A copy is returned, preventing direct changes to the collection
        """
        return sorted(self._items.values(), key = lambda item: item.id)

    @items.setter
    def items(self, val: Collection[Identified]):
        if isinstance(val, dict):
            self._items = val.copy()
        elif isinstance(val, Collection) and not isinstance(val, str):
            self._items = {item.id: item for item in val}
        elif val is None:
            self._items = dict()
        else:
            raise ValueError(f"Items must be supplied as a valid Collection, not {type(val)}")

    def add(self, id: int) -> Identified:
        """
        Create a new Identified instance and append it to the collection

        :param id: a valid Identified ID number
        :returns: a new Identified instance
        """
        item = Identified(id = id)

        self.append(item)

        return item

    def append(self, item: Type[Identified]):
        """
        Append an item to the collection

        :param item: the object to append to the collection
        """
        if not self.__contains__(item):
            self._items[item.id] = item
        else:
            raise ValueError(f"An item with ID #{item.id} already exists")

    def exists(self, id: int) -> bool:
        """
        Check if an item with the specified ID number exists in the item collection

        :param id: a valid Identified id number
        :returns: True if an item is found with matching ID
        """
        return id in self._items.keys()

    def update(self, items: Union[Identified, Iterable[Identified]]):
        """
        Replace existing items in the collection by matching ID numbers.
        If the item does not already exist in the collection, it is appended.

        :param: a single, or sequence of, Identified instances
        """
        items = list(items)

        for item in items:
            self._items[item.id] = item

    def __contains__(self, item: Identified) -> bool:
        """
        Check if an item exists in the item collection

        :param item: an instance of Identified
        :returns: True if an item is found with matching ID
        """
        return item.id in self._items

    def __delitem__(self, id: int) -> Identified:
        """
        Remove an item from the collection and return it if it can be
        found in the collection

        :param id: the id of the item to remove
        :returns: the item in the collection with matching ID
        """
        return self._items.pop(id)

    def __getitem__(self, id: int) -> Identified:
        """
        Returns an item with the specified ID number from the item collection

        :param id: a valid Identified ID number
        :returns: an item from the collection, if found
        """
        if not Identified._id_is_valid(id):
            raise TypeError(f"ID must be of type {type(int)}, not {type(id)}")
        if id not in self._items:
            raise KeyError(f"An item with ID {id} could not be found in the collection")

        return self._items[id]

    def __iter__(self):
        yield from self.items

    def __len__(self) -> int:
        return len(self._items)

    def __setitem__(self, id: int, item: Identified):
        if not isinstance(item, Identified):
            raise TypeError(f"Item must be of type {type(Identified)}, not {type(item)}")
        if not self.__contains__(item):
            raise KeyError(f"An item with ID {item.id} could not be found in the collection")
        self._items[item.id] = item

    def __reversed__(self) -> List[Identified]:
        return reversed(self.items)


class Named:
    """
    An object with a string identifier
    """
    def __init__(self, name: str):
        self.name = name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val: str):
        self._name = str(val)


class Unique(Identified):
    """
    An object with a auto incremented integer ID number
    """
    id_count = 0

    def __init__(self):
        self._id = self.id_increment()

    @Identified.id.setter
    def id(self, val: int):
        """
        Overrides base method to make id read-only
        """
        pass

    def id_increment(self) -> int:
        """
        Increments the built-in ID counter

        :returns: the auto incremented ID number
        """
        Unique.id_count += 1

        return Unique.id_count


class TimeStamped:
    def __init__(self, timestamp: datetime = None):
        if timestamp is None:
            timestamp = datetime.now()

        self.timestamp = timestamp

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    @timestamp.setter
    def timestamp(self, val: datetime):
        self._timestamp = val


class TimeStampElapsed(TimeStamped):
    def __init__(self, timestamp: datetime = None, timestamp_initial: datetime = None):
        if timestamp is None:
            timestamp = datetime.now()
        if timestamp_initial is None:
            timestamp_initial = timestamp

        self._timestamp = timestamp
        self.timestamp_initial = timestamp_initial

    @property
    def timestamp_elapsed(self) -> timedelta:
        return self.timestamp - self.timestamp_initial

    @property
    def timestamp_elapsed_hours(self) -> float:
        return (self.timestamp_elapsed_seconds / 60) / 60

    @property
    def timestamp_elapsed_minutes(self) -> int:
        return int(self.timestamp_elapsed_seconds / 60)

    @property
    def timestamp_elapsed_seconds(self) -> int:
        return self.timestamp_elapsed.total_seconds()

    @property
    def timestamp_initial(self) -> datetime:
        return self._timestamp_initial

    @timestamp_initial.setter
    def timestamp_initial(self, val: datetime):
        self._timestamp_initial = val